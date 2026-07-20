from pathlib import Path

import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="MGF Feature Filter",
    page_icon="🧪",
    layout="wide",
)

st.title("MGF Feature Filter")
st.caption(
    "Filtre um arquivo MGF usando IDs presentes em uma tabela CSV, TSV ou TXT "
    "contendo VIP scores, correlações, p-valores ou qualquer outra métrica."
)


# ============================================================
# Funções auxiliares
# ============================================================

def normalize_id(value) -> str:
    """
    Normaliza IDs para comparação.

    Exemplos:
    69 -> '69'
    69.0 -> '69'
    ' 069 ' -> '069'
    '69/150.0260mz/0.04min' -> '69/150.0260mz/0.04min'
    """
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]

    return text


def extract_first_token(text: str) -> str:
    """
    Extrai o primeiro identificador de uma string.

    Exemplos:
    '69/150.0260mz/0.04min' -> '69'
    '69 150.0260 0.04' -> '69'
    '69' -> '69'
    """
    text = str(text).strip()

    if "/" in text:
        return normalize_id(text.split("/", 1)[0])

    if text:
        return normalize_id(text.split()[0])

    return ""


def read_table(uploaded_file) -> pd.DataFrame:
    """
    Lê CSV/TSV/TXT tentando detectar automaticamente o separador.
    """
    raw = uploaded_file.getvalue()

    encodings = ["utf-8-sig", "utf-8", "latin-1"]
    last_error = None

    for encoding in encodings:
        try:
            text = raw.decode(encoding)
            return pd.read_csv(
                io.StringIO(text),
                sep=None,
                engine="python",
            )
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Não foi possível ler a tabela: {last_error}")


def parse_mgf(mgf_text: str) -> list[dict]:
    """
    Divide o MGF em blocos BEGIN IONS / END IONS.
    Preserva o texto original de cada bloco.
    """
    spectra = []
    current_block = []
    inside_block = False

    for line in mgf_text.splitlines():
        stripped = line.strip()

        if stripped.upper() == "BEGIN IONS":
            inside_block = True
            current_block = [line]
            continue

        if inside_block:
            current_block.append(line)

            if stripped.upper() == "END IONS":
                block_text = "\n".join(current_block)
                metadata = parse_metadata(current_block)

                spectra.append(
                    {
                        "block": block_text,
                        "metadata": metadata,
                    }
                )

                current_block = []
                inside_block = False

    return spectra


def parse_metadata(block_lines: list[str]) -> dict:
    """
    Lê metadados do bloco MGF.
    """
    metadata = {}

    for line in block_lines:
        stripped = line.strip()

        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        metadata[key.strip().upper()] = value.strip()

    return metadata


def get_mgf_feature_id(metadata: dict, source_field: str, extraction_mode: str) -> str:
    """
    Obtém o ID de um espectro a partir do campo selecionado.
    """
    raw_value = metadata.get(source_field.upper(), "")

    if extraction_mode == "Valor completo":
        return normalize_id(raw_value)

    if extraction_mode == "Primeiro valor antes de '/'":
        return normalize_id(str(raw_value).split("/", 1)[0])

    if extraction_mode == "Primeiro token":
        return extract_first_token(raw_value)

    if extraction_mode == "Número de SCANS":
        return normalize_id(metadata.get("SCANS", ""))

    return normalize_id(raw_value)


def build_filtered_mgf(spectra: list[dict], selected_ids: set[str], source_field: str, extraction_mode: str):
    """
    Retorna os blocos selecionados e um resumo de correspondência.
    """
    kept_blocks = []
    mgf_ids = []
    matched_ids = set()

    for spectrum in spectra:
        feature_id = get_mgf_feature_id(
            spectrum["metadata"],
            source_field,
            extraction_mode,
        )

        mgf_ids.append(feature_id)

        if feature_id in selected_ids:
            kept_blocks.append(spectrum["block"])
            matched_ids.add(feature_id)

    filtered_text = "\n\n".join(kept_blocks)

    if filtered_text:
        filtered_text += "\n"

    return filtered_text, mgf_ids, matched_ids


def make_summary_table(spectra: list[dict], source_field: str, extraction_mode: str) -> pd.DataFrame:
    rows = []

    for index, spectrum in enumerate(spectra, start=1):
        metadata = spectrum["metadata"]

        rows.append(
            {
                "Spectrum": index,
                "Extracted_ID": get_mgf_feature_id(
                    metadata,
                    source_field,
                    extraction_mode,
                ),
                "TITLE": metadata.get("TITLE", ""),
                "SCANS": metadata.get("SCANS", ""),
                "PEPMASS": metadata.get("PEPMASS", ""),
                "RTINSECONDS": metadata.get("RTINSECONDS", ""),
                "CHARGE": metadata.get("CHARGE", ""),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Uploads
# ============================================================

col1, col2 = st.columns(2)

with col1:
    mgf_file = st.file_uploader(
        "1. Importe o arquivo MGF",
        type=["mgf"],
    )

with col2:
    table_file = st.file_uploader(
        "2. Importe a tabela de seleção",
        type=["csv", "tsv", "txt"],
        help="A tabela deve conter pelo menos uma coluna com os IDs dos features.",
    )


if not mgf_file or not table_file:
    st.info("Importe os dois arquivos para configurar a filtragem.")
    st.stop()


# ============================================================
# Leitura dos arquivos
# ============================================================

try:
    mgf_text = mgf_file.getvalue().decode("utf-8-sig")
except UnicodeDecodeError:
    mgf_text = mgf_file.getvalue().decode("latin-1")

try:
    table_df = read_table(table_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

spectra = parse_mgf(mgf_text)

if not spectra:
    st.error("Nenhum bloco BEGIN IONS / END IONS foi encontrado no MGF.")
    st.stop()

if table_df.empty:
    st.error("A tabela importada está vazia.")
    st.stop()


# ============================================================
# Configuração
# ============================================================

st.subheader("Configuração da correspondência")

config_col1, config_col2, config_col3 = st.columns(3)

with config_col1:
    id_column = st.selectbox(
        "Coluna de ID na tabela",
        options=list(table_df.columns),
    )

with config_col2:
    mgf_source_field = st.selectbox(
        "Campo do MGF contendo o ID",
        options=["TITLE", "SCANS", "FEATURE_ID"],
        index=0,
        help=(
            "Na maior parte dos arquivos exportados pelo MZmine/GNPS, "
            "o ID aparece em TITLE ou SCANS."
        ),
    )

with config_col3:
    extraction_mode = st.selectbox(
        "Como extrair o ID do MGF",
        options=[
            "Primeiro valor antes de '/'",
            "Valor completo",
            "Primeiro token",
            "Número de SCANS",
        ],
        index=0,
        help=(
            "Exemplo: TITLE=69/150.0260mz/0.04min → ID 69."
        ),
    )


st.subheader("Critério de seleção")

selection_mode = st.radio(
    "Selecione como os IDs serão escolhidos",
    options=[
        "Usar todos os IDs da tabela",
        "Aplicar corte por valor",
        "Selecionar Top N",
    ],
    horizontal=True,
)

working_df = table_df.copy()
working_df["_normalized_id"] = working_df[id_column].map(normalize_id)
working_df = working_df[working_df["_normalized_id"] != ""].copy()

value_column = None
ascending = False

if selection_mode in ["Aplicar corte por valor", "Selecionar Top N"]:
    numeric_candidates = []

    for column in table_df.columns:
        converted = pd.to_numeric(table_df[column], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_candidates.append(column)

    if not numeric_candidates:
        st.error("Nenhuma coluna numérica foi encontrada para aplicar o critério selecionado.")
        st.stop()

    value_column = st.selectbox(
        "Coluna numérica usada para seleção",
        options=numeric_candidates,
    )

    working_df["_numeric_value"] = pd.to_numeric(
        working_df[value_column],
        errors="coerce",
    )
    working_df = working_df.dropna(subset=["_numeric_value"])

if selection_mode == "Aplicar corte por valor":
    operator = st.selectbox(
        "Operador",
        options=[">=", ">", "<=", "<", "valor absoluto >="],
    )

    default_cutoff = float(working_df["_numeric_value"].median()) if not working_df.empty else 0.0

    cutoff = st.number_input(
        "Valor de corte",
        value=default_cutoff,
        format="%.6f",
    )

    if operator == ">=":
        selected_df = working_df[working_df["_numeric_value"] >= cutoff]
    elif operator == ">":
        selected_df = working_df[working_df["_numeric_value"] > cutoff]
    elif operator == "<=":
        selected_df = working_df[working_df["_numeric_value"] <= cutoff]
    elif operator == "<":
        selected_df = working_df[working_df["_numeric_value"] < cutoff]
    else:
        selected_df = working_df[working_df["_numeric_value"].abs() >= cutoff]

elif selection_mode == "Selecionar Top N":
    rank_direction = st.selectbox(
        "Ordenação",
        options=[
            "Maiores valores",
            "Menores valores",
            "Maiores valores absolutos",
        ],
    )

    max_n = max(1, len(working_df))

    top_n = st.number_input(
        "Número de features",
        min_value=1,
        max_value=max_n,
        value=min(30, max_n),
        step=1,
    )

    if rank_direction == "Maiores valores":
        selected_df = working_df.nlargest(int(top_n), "_numeric_value")
    elif rank_direction == "Menores valores":
        selected_df = working_df.nsmallest(int(top_n), "_numeric_value")
    else:
        selected_df = (
            working_df.assign(_abs_value=working_df["_numeric_value"].abs())
            .nlargest(int(top_n), "_abs_value")
        )

else:
    selected_df = working_df.copy()


selected_ids = set(selected_df["_normalized_id"].tolist())


# ============================================================
# Filtragem
# ============================================================

filtered_mgf, mgf_ids, matched_ids = build_filtered_mgf(
    spectra,
    selected_ids,
    mgf_source_field,
    extraction_mode,
)

mgf_id_set = {item for item in mgf_ids if item}
unmatched_table_ids = selected_ids - mgf_id_set
unselected_mgf_ids = mgf_id_set - selected_ids

summary_df = make_summary_table(
    spectra,
    mgf_source_field,
    extraction_mode,
)

summary_df["Selected"] = summary_df["Extracted_ID"].isin(selected_ids)


# ============================================================
# Resultados
# ============================================================

st.subheader("Resultado")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Espectros no MGF", len(spectra))
metric2.metric("IDs selecionados", len(selected_ids))
metric3.metric("Espectros exportados", summary_df["Selected"].sum())
metric4.metric("IDs sem correspondência", len(unmatched_table_ids))

if len(selected_ids) == 0:
    st.warning("Nenhum ID foi selecionado com o critério atual.")

elif summary_df["Selected"].sum() == 0:
    st.error(
        "Nenhum espectro foi encontrado. Verifique a coluna de ID, "
        "o campo do MGF e o método de extração."
    )

else:
    st.success(
        f"{int(summary_df['Selected'].sum())} espectros foram encontrados e serão exportados."
    )


tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Tabela importada",
        "Espectros do MGF",
        "IDs sem correspondência",
        "Prévia do MGF filtrado",
    ]
)

with tab1:
    display_columns = [
        column
        for column in selected_df.columns
        if not column.startswith("_")
    ]

    st.dataframe(
        selected_df[display_columns],
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

with tab3:
    if unmatched_table_ids:
        unmatched_df = pd.DataFrame(
            {"Unmatched_table_ID": sorted(unmatched_table_ids)}
        )
        st.dataframe(
            unmatched_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("Todos os IDs selecionados foram encontrados no MGF.")

with tab4:
    if filtered_mgf:
        preview_limit = 15000
        preview = filtered_mgf[:preview_limit]

        st.code(preview, language="text")

        if len(filtered_mgf) > preview_limit:
            st.caption("Prévia truncada para manter o aplicativo responsivo.")
    else:
        st.info("Nenhum conteúdo disponível para prévia.")


# ============================================================
# Downloads
# ============================================================

st.subheader("Exportação")

original_stem = Path(mgf_file.name).stem
output_name = f"{original_stem}_filtered.mgf"

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        "Baixar MGF filtrado",
        data=filtered_mgf.encode("utf-8"),
        file_name=output_name,
        mime="text/plain",
        disabled=not bool(filtered_mgf),
        use_container_width=True,
    )

with download_col2:
    selected_export = selected_df[
        [column for column in selected_df.columns if not column.startswith("_")]
    ].copy()

    st.download_button(
        "Baixar tabela selecionada",
        data=selected_export.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{Path(table_file.name).stem}_selected.csv",
        mime="text/csv",
        disabled=selected_export.empty,
        use_container_width=True,
    )

with download_col3:
    unmatched_export = pd.DataFrame(
        {"Unmatched_table_ID": sorted(unmatched_table_ids)}
    )

    st.download_button(
        "Baixar IDs não encontrados",
        data=unmatched_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="unmatched_ids.csv",
        mime="text/csv",
        disabled=unmatched_export.empty,
        use_container_width=True,
    )

with st.expander("Como o aplicativo identifica os features"):
    st.markdown(
        """
O aplicativo compara uma coluna da tabela com um identificador extraído de cada espectro do MGF.

Exemplo:

```text
TITLE=69/150.0260mz/0.04min
```

Usando **Primeiro valor antes de '/'**, o identificador será:

```text
69
```

Assim, uma linha da tabela contendo `69` selecionará esse espectro.

Também é possível usar:

- o valor completo de `TITLE`;
- o primeiro token de `TITLE`;
- o campo `SCANS`;
- o campo `FEATURE_ID`, quando presente.
"""
    )
