import io
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="MGF Feature Filter",
    page_icon="🧪",
    layout="wide",
)

# ============================================================
# Load the logo LAABio
logo_LAABio = Image.open("static/LAABio.png")
# Display the logo in the sidebar or header
st.image(logo_LAABio, width=300)


st.title("MGF Feature Filter")
st.caption(
    "Filtre uma tabela por qualquer coluna, use os IDs resultantes para selecionar "
    "espectros em um arquivo MGF e exporte os arquivos filtrados."
)



# ============================================================
# Funções auxiliares
# ============================================================

def normalize_id(value) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]

    return text


def extract_first_token(text: str) -> str:
    text = str(text).strip()

    if "/" in text:
        return normalize_id(text.split("/", 1)[0])

    if text:
        return normalize_id(text.split()[0])

    return ""


def read_table(uploaded_file) -> pd.DataFrame:
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


def parse_metadata(block_lines: list[str]) -> dict:
    metadata = {}

    for line in block_lines:
        stripped = line.strip()

        if "=" not in stripped:
            continue

        key, value = stripped.split("=", 1)
        metadata[key.strip().upper()] = value.strip()

    return metadata


def parse_mgf(mgf_text: str) -> list[dict]:
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
                spectra.append(
                    {
                        "block": "\n".join(current_block),
                        "metadata": parse_metadata(current_block),
                    }
                )
                current_block = []
                inside_block = False

    return spectra


def transform_table_id(value, mode: str) -> str:
    value = normalize_id(value)

    if mode == "Valor completo":
        return value

    if mode == "Primeiro valor antes de '/'":
        return normalize_id(value.split("/", 1)[0])

    if mode == "Primeiro token":
        return extract_first_token(value)

    return value


def get_mgf_feature_id(metadata: dict, source_field: str, extraction_mode: str) -> str:
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


def is_numeric_series(series: pd.Series) -> bool:
    converted = pd.to_numeric(series, errors="coerce")
    return converted.notna().sum() >= max(1, int(len(series) * 0.5))


def apply_numeric_filter(
    df: pd.DataFrame,
    column: str,
    operator: str,
    value1: float,
    value2: float | None = None,
) -> pd.DataFrame:
    numeric = pd.to_numeric(df[column], errors="coerce")

    if operator == ">":
        mask = numeric > value1
    elif operator == ">=":
        mask = numeric >= value1
    elif operator == "<":
        mask = numeric < value1
    elif operator == "<=":
        mask = numeric <= value1
    elif operator == "=":
        mask = numeric == value1
    elif operator == "!=":
        mask = numeric != value1
    elif operator == "Entre":
        low = min(value1, value2)
        high = max(value1, value2)
        mask = numeric.between(low, high, inclusive="both")
    elif operator == "Fora do intervalo":
        low = min(value1, value2)
        high = max(value1, value2)
        mask = ~numeric.between(low, high, inclusive="both")
    elif operator == "Valor absoluto >=":
        mask = numeric.abs() >= value1
    elif operator == "Valor absoluto <=":
        mask = numeric.abs() <= value1
    else:
        mask = pd.Series(True, index=df.index)

    return df[mask.fillna(False)].copy()


def apply_text_filter(
    df: pd.DataFrame,
    column: str,
    operator: str,
    values: list[str] | None = None,
    text_value: str = "",
    case_sensitive: bool = False,
) -> pd.DataFrame:
    series = df[column].fillna("").astype(str)

    if not case_sensitive:
        comparison_series = series.str.lower()
        comparison_text = text_value.lower()
        comparison_values = [str(v).lower() for v in (values or [])]
    else:
        comparison_series = series
        comparison_text = text_value
        comparison_values = [str(v) for v in (values or [])]

    if operator == "É um dos valores":
        mask = comparison_series.isin(comparison_values)
    elif operator == "Não é um dos valores":
        mask = ~comparison_series.isin(comparison_values)
    elif operator == "Contém":
        mask = comparison_series.str.contains(
            re.escape(comparison_text),
            na=False,
            regex=True,
        )
    elif operator == "Não contém":
        mask = ~comparison_series.str.contains(
            re.escape(comparison_text),
            na=False,
            regex=True,
        )
    elif operator == "Começa com":
        mask = comparison_series.str.startswith(comparison_text, na=False)
    elif operator == "Termina com":
        mask = comparison_series.str.endswith(comparison_text, na=False)
    elif operator == "Igual a":
        mask = comparison_series == comparison_text
    elif operator == "Diferente de":
        mask = comparison_series != comparison_text
    elif operator == "Está vazio":
        mask = series.str.strip() == ""
    elif operator == "Não está vazio":
        mask = series.str.strip() != ""
    else:
        mask = pd.Series(True, index=df.index)

    return df[mask].copy()


def build_filtered_mgf(
    spectra: list[dict],
    selected_ids: set[str],
    source_field: str,
    extraction_mode: str,
):
    kept_blocks = []
    mgf_ids = []

    for spectrum in spectra:
        feature_id = get_mgf_feature_id(
            spectrum["metadata"],
            source_field,
            extraction_mode,
        )

        mgf_ids.append(feature_id)

        if feature_id in selected_ids:
            kept_blocks.append(spectrum["block"])

    filtered_text = "\n\n".join(kept_blocks)

    if filtered_text:
        filtered_text += "\n"

    return filtered_text, mgf_ids


def make_summary_table(
    spectra: list[dict],
    source_field: str,
    extraction_mode: str,
) -> pd.DataFrame:
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
                "FEATURE_ID": metadata.get("FEATURE_ID", ""),
                "PEPMASS": metadata.get("PEPMASS", ""),
                "RTINSECONDS": metadata.get("RTINSECONDS", ""),
                "CHARGE": metadata.get("CHARGE", ""),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# Uploads
# ============================================================

upload_col1, upload_col2 = st.columns(2)

with upload_col1:
    mgf_file = st.file_uploader(
        "1. Importe o arquivo MGF",
        type=["mgf"],
    )

with upload_col2:
    table_file = st.file_uploader(
        "2. Importe a tabela",
        type=["csv", "tsv", "txt"],
        help="A tabela deve conter uma coluna que possa ser relacionada aos IDs do MGF.",
    )

if not table_file:
    st.info("Importe pelo menos uma tabela para configurar os filtros.")
    st.stop()


# ============================================================
# Leitura da tabela
# ============================================================

try:
    table_df = read_table(table_file)
except Exception as exc:
    st.error(str(exc))
    st.stop()

if table_df.empty:
    st.error("A tabela importada está vazia.")
    st.stop()

st.subheader("Tabela importada")

metric_col1, metric_col2 = st.columns(2)
metric_col1.metric("Linhas", len(table_df))
metric_col2.metric("Colunas", len(table_df.columns))

with st.expander("Visualizar tabela original", expanded=True):
    st.dataframe(
        table_df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Filtros da tabela
# ============================================================

st.subheader("Filtrar a tabela")

st.caption(
    "Você pode combinar vários filtros. Os filtros são aplicados em sequência, "
    "equivalentes a uma condição AND."
)

number_of_filters = st.number_input(
    "Número de filtros",
    min_value=0,
    max_value=10,
    value=0,
    step=1,
)

filtered_df = table_df.copy()
filter_descriptions = []

for filter_index in range(int(number_of_filters)):
    with st.expander(
        f"Filtro {filter_index + 1}",
        expanded=True,
    ):
        filter_columns = list(table_df.columns)
        default_filter_index = (
            filter_columns.index("VIP")
            if "VIP" in filter_columns
            else 0
        )

        column = st.selectbox(
            "Coluna",
            options=filter_columns,
            index=default_filter_index,
            key=f"filter_column_{filter_index}",
        )

        numeric_column = is_numeric_series(table_df[column])

        if numeric_column:
            operator = st.selectbox(
                "Condição",
                options=[
                    ">",
                    ">=",
                    "<",
                    "<=",
                    "=",
                    "!=",
                    "Entre",
                    "Fora do intervalo",
                    "Valor absoluto >=",
                    "Valor absoluto <=",
                ],
                index=0,
                key=f"numeric_operator_{filter_index}",
            )

            numeric_values = pd.to_numeric(
                table_df[column],
                errors="coerce",
            ).dropna()

            default_value = 2.0

            value_col1, value_col2 = st.columns(2)

            with value_col1:
                value1 = st.number_input(
                    "Valor",
                    value=default_value,
                    format="%.8f",
                    key=f"numeric_value1_{filter_index}",
                )

            value2 = None

            if operator in ["Entre", "Fora do intervalo"]:
                with value_col2:
                    value2 = st.number_input(
                        "Segundo valor",
                        value=(
                            float(numeric_values.max())
                            if not numeric_values.empty
                            else default_value
                        ),
                        format="%.8f",
                        key=f"numeric_value2_{filter_index}",
                    )

            filtered_df = apply_numeric_filter(
                filtered_df,
                column,
                operator,
                value1,
                value2,
            )

            if value2 is None:
                filter_descriptions.append(
                    f"{column} {operator} {value1}"
                )
            else:
                filter_descriptions.append(
                    f"{column} {operator} {value1} e {value2}"
                )

        else:
            unique_values = (
                table_df[column]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .tolist()
            )

            text_operator = st.selectbox(
                "Condição",
                options=[
                    "É um dos valores",
                    "Não é um dos valores",
                    "Contém",
                    "Não contém",
                    "Começa com",
                    "Termina com",
                    "Igual a",
                    "Diferente de",
                    "Está vazio",
                    "Não está vazio",
                ],
                key=f"text_operator_{filter_index}",
            )

            selected_values = []
            text_value = ""

            if text_operator in [
                "É um dos valores",
                "Não é um dos valores",
            ]:
                selected_values = st.multiselect(
                    "Valores",
                    options=unique_values,
                    key=f"text_values_{filter_index}",
                )

            elif text_operator not in [
                "Está vazio",
                "Não está vazio",
            ]:
                text_value = st.text_input(
                    "Texto",
                    key=f"text_value_{filter_index}",
                )

            case_sensitive = st.checkbox(
                "Diferenciar maiúsculas de minúsculas",
                value=False,
                key=f"case_sensitive_{filter_index}",
            )

            filtered_df = apply_text_filter(
                filtered_df,
                column,
                text_operator,
                values=selected_values,
                text_value=text_value,
                case_sensitive=case_sensitive,
            )

            if selected_values:
                filter_descriptions.append(
                    f"{column}: {text_operator} {selected_values}"
                )
            else:
                filter_descriptions.append(
                    f"{column}: {text_operator} {text_value}"
                )


result_col1, result_col2 = st.columns(2)
result_col1.metric("Linhas originais", len(table_df))
result_col2.metric("Linhas após os filtros", len(filtered_df))

if filter_descriptions:
    st.caption("Filtros aplicados: " + " | ".join(filter_descriptions))

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
)

st.download_button(
    "Baixar CSV filtrado",
    data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
    file_name=f"{Path(table_file.name).stem}_filtered.csv",
    mime="text/csv",
    disabled=filtered_df.empty,
)


# ============================================================
# Configuração dos IDs
# ============================================================

st.subheader("Selecionar IDs para o MGF")

id_col1, id_col2 = st.columns(2)

with id_col1:
    id_column = st.selectbox(
        "Coluna da tabela que identifica os espectros",
        options=list(filtered_df.columns),
        index=(
            list(filtered_df.columns).index("feature")
            if "feature" in filtered_df.columns
            else 0
        ),
    )

with id_col2:
    table_id_mode = st.selectbox(
        "Como extrair o ID da coluna da tabela",
        options=[
            "Valor completo",
            "Primeiro valor antes de '/'",
            "Primeiro token",
        ],
        index=1,
        help=(
            "No exemplo '619/123.9643mz/0.21min', a opção "
            "'Primeiro valor antes de /' produz o ID 619."
        ),
    )

working_df = filtered_df.copy()
working_df["_selected_id"] = working_df[id_column].map(
    lambda value: transform_table_id(value, table_id_mode)
)
working_df = working_df[working_df["_selected_id"] != ""].copy()

selected_ids = set(working_df["_selected_id"])

st.metric("IDs únicos selecionados", len(selected_ids))

with st.expander("Visualizar IDs selecionados"):
    st.dataframe(
        pd.DataFrame({"Selected_ID": sorted(selected_ids)}),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# MGF
# ============================================================

if not mgf_file:
    st.info(
        "O CSV filtrado já pode ser exportado. Importe também um MGF "
        "para selecionar e exportar os espectros correspondentes."
    )
    st.stop()

try:
    mgf_text = mgf_file.getvalue().decode("utf-8-sig")
except UnicodeDecodeError:
    mgf_text = mgf_file.getvalue().decode("latin-1")

spectra = parse_mgf(mgf_text)

if not spectra:
    st.error("Nenhum bloco BEGIN IONS / END IONS foi encontrado no MGF.")
    st.stop()

st.subheader("Correspondência com o MGF")

mgf_col1, mgf_col2 = st.columns(2)

with mgf_col1:
    mgf_source_field = st.selectbox(
        "Campo do MGF que contém o ID",
        options=["TITLE", "SCANS", "FEATURE_ID"],
        index=2,
    )

with mgf_col2:
    mgf_extraction_mode = st.selectbox(
        "Como extrair o ID do MGF",
        options=[
            "Valor completo",
            "Primeiro valor antes de '/'",
            "Primeiro token",
            "Número de SCANS",
        ],
        index=1,
    )

filtered_mgf, mgf_ids = build_filtered_mgf(
    spectra,
    selected_ids,
    mgf_source_field,
    mgf_extraction_mode,
)

summary_df = make_summary_table(
    spectra,
    mgf_source_field,
    mgf_extraction_mode,
)

summary_df["Selected"] = summary_df["Extracted_ID"].isin(selected_ids)

mgf_id_set = {
    normalize_id(item)
    for item in mgf_ids
    if normalize_id(item)
}

unmatched_table_ids = selected_ids - mgf_id_set

metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Espectros no MGF", len(spectra))
metric2.metric("IDs da tabela", len(selected_ids))
metric3.metric(
    "Espectros selecionados",
    int(summary_df["Selected"].sum()),
)
metric4.metric(
    "IDs não encontrados",
    len(unmatched_table_ids),
)

if selected_ids and summary_df["Selected"].sum() == 0:
    st.error(
        "Nenhuma correspondência foi encontrada. Verifique a coluna de ID, "
        "o campo do MGF e as formas de extração dos IDs."
    )
elif summary_df["Selected"].sum() > 0:
    st.success(
        f"{int(summary_df['Selected'].sum())} espectros serão exportados."
    )

tab1, tab2, tab3 = st.tabs(
    [
        "Espectros do MGF",
        "IDs não encontrados",
        "Prévia do MGF filtrado",
    ]
)

with tab1:
    st.dataframe(
        summary_df,
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    if unmatched_table_ids:
        st.dataframe(
            pd.DataFrame(
                {
                    "Unmatched_table_ID": sorted(
                        unmatched_table_ids
                    )
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("Todos os IDs selecionados foram encontrados.")

with tab3:
    if filtered_mgf:
        preview_limit = 15000
        st.code(
            filtered_mgf[:preview_limit],
            language="text",
        )

        if len(filtered_mgf) > preview_limit:
            st.caption("Prévia truncada.")
    else:
        st.info("Nenhum espectro disponível para prévia.")


# ============================================================
# Downloads finais
# ============================================================

st.subheader("Exportação")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        "Baixar MGF filtrado",
        data=filtered_mgf.encode("utf-8"),
        file_name=f"{Path(mgf_file.name).stem}_filtered.mgf",
        mime="text/plain",
        disabled=not bool(filtered_mgf),
        use_container_width=True,
    )

with download_col2:
    st.download_button(
        "Baixar tabela usada no MGF",
        data=filtered_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{Path(table_file.name).stem}_used_for_mgf.csv",
        mime="text/csv",
        disabled=filtered_df.empty,
        use_container_width=True,
    )

with download_col3:
    unmatched_df = pd.DataFrame(
        {"Unmatched_table_ID": sorted(unmatched_table_ids)}
    )

    st.download_button(
        "Baixar IDs não encontrados",
        data=unmatched_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="unmatched_ids.csv",
        mime="text/csv",
        disabled=unmatched_df.empty,
        use_container_width=True,
    )


with st.expander("Exemplo com o arquivo plsda_vip.csv"):
    st.markdown(
        """
Para o arquivo fornecido:

- **Coluna de ID:** `feature`
- **Forma de extração do ID da tabela:** `Primeiro valor antes de '/'`
- **Coluna de filtro:** `VIP`
- **Exemplo de condição:** `VIP >= 1.0`
- **Campo do MGF:** normalmente `TITLE` ou `SCANS`
- **Forma de extração do ID do MGF:** deve corresponder ao formato usado na tabela

Exemplo de feature:

```text
619/123.9643mz/0.21min
```

O ID extraído será:

```text
619
```
"""
    )
