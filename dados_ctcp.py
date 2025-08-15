from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ------------------------------
# Config da página
# ------------------------------
st.set_page_config(
    page_title="Análise CTCP – Dashboard",
    page_icon="🚌",
    layout="wide"
)

# ------------------------------
# Utilitários
# ------------------------------
ROOT = Path(__file__).parent.resolve()
DEFAULTS = [
    ROOT / "dados_ctcp.csv",
    ROOT / "data" / "dados_ctcp.csv",
]

def find_default_csv() -> Path | None:
    for p in DEFAULTS:
        if p.exists():
            return p
    return None

@st.cache_data(show_spinner=False)
def read_csv_any(path_or_buffer, sep=None, encoding="utf-8"):
    """
    Lê CSV com tolerância:
    - tenta separador informado (ou autodetecta)
    - tenta UTF-8; se falhar, tenta latin-1
    """
    try:
        return pd.read_csv(path_or_buffer, sep=sep, encoding=encoding)
    except Exception:
        # fallback
        try:
            return pd.read_csv(path_or_buffer, sep=sep, encoding="latin-1")
        except Exception as e:
            raise e

def is_number_dtype(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def is_cat_dtype(s: pd.Series) -> bool:
    # "categoria" aqui = texto/objeto/category
    return pd.api.types.is_categorical_dtype(s) or pd.api.types.is_object_dtype(s)

def plot_hbar(series: pd.Series, title: str, xlabel: str = "Quantidade"):
    series = series.dropna()
    if series.empty:
        st.info("Sem dados para o gráfico.")
        return

    series = series.sort_values(ascending=True)
    fig_height = max(3, 0.35 * len(series))
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(series.index.astype(str), series.values)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    # anota valores
    for i, v in enumerate(series.values):
        ax.text(v, i, f" {v}", va="center")

    st.pyplot(fig, clear_figure=True)

def plot_hist(series: pd.Series, title: str, bins: int = 20):
    series = series.dropna()
    if series.empty:
        st.info("Sem dados para o gráfico.")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(series.values, bins=bins)
    ax.set_title(title)
    ax.set_xlabel(series.name)
    ax.set_ylabel("Frequência")
    st.pyplot(fig, clear_figure=True)

# ------------------------------
# Sidebar — entrada de dados
# ------------------------------
st.sidebar.header("🔧 Configuração de dados")

uploaded = st.sidebar.file_uploader("Carregue um CSV", type=["csv"])

sep_choice = st.sidebar.selectbox(
    "Separador (opcional)",
    options=["(auto)", ",", ";", "|", "\t"],
    index=0
)
sep = None if sep_choice == "(auto)" else sep_choice

default_path = find_default_csv()
use_default = st.sidebar.checkbox(
    f"Usar CSV do repositório ({default_path.name})" if default_path else "Usar CSV do repositório",
    value=bool(default_path is not None)
) if default_path else False

# ------------------------------
# Carregamento dos dados
# ------------------------------
df: pd.DataFrame | None = None

try:
    if uploaded is not None:
        df = read_csv_any(uploaded, sep=sep)
    elif use_default and default_path:
        df = read_csv_any(default_path, sep=sep)
    else:
        st.warning("Carregue um CSV na barra lateral ou inclua `dados_ctcp.csv` no repositório.")
        st.stop()
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}")
    st.stop()

# ------------------------------
# Cabeçalho
# ------------------------------
st.title("Análise da Frota CTCP – Pelotas (RS)")
st.caption("Dashboard genérico para exploração de dados CSV (Streamlit).")

# ------------------------------
# Visão geral
# ------------------------------
st.subheader("📄 Amostra dos dados")
st.write(df.head(20))

col_a, col_b, col_c = st.columns([1, 1, 1])
with col_a:
    st.metric("Linhas", f"{len(df):,}".replace(",", "."))
with col_b:
    st.metric("Colunas", df.shape[1])
with col_c:
    st.metric("Valores ausentes", int(df.isna().sum().sum()))

with st.expander("Dicionário de dados (tipos)"):
    info = pd.DataFrame({
        "coluna": df.columns,
        "dtype": [str(t) for t in df.dtypes],
        "n_null": df.isna().sum().values
    })
    st.dataframe(info, use_container_width=True)

# ------------------------------
# Filtros rápidos (opcionais)
# ------------------------------
st.subheader("🎛️ Filtros rápidos")
cat_cols = [c for c in df.columns if is_cat_dtype(df[c])]
num_cols = [c for c in df.columns if is_number_dtype(df[c])]

with st.expander("Aplicar filtros por coluna (categorias)"):
    for c in cat_cols[:10]:  # limita 10 para não poluir a UI
        vals = sorted(df[c].dropna().unique().tolist())
        if len(vals) > 0 and len(vals) <= 200:
            selected = st.multiselect(f"{c}", vals, key=f"f_{c}")
            if selected:
                df = df[df[c].isin(selected)]

# ------------------------------
# Gráficos
# ------------------------------
st.subheader("📊 Gráficos")

# Gráfico de barras (categorias)
with st.container():
    st.markdown("**Contagem por categoria**")
    if cat_cols:
        cat_col = st.selectbox("Coluna categórica", cat_cols, index=0, key="cat_for_bar")
        top_n = st.slider("Top N", min_value=5, max_value=50, value=min(15, len(df[cat_col].unique())))
        counts = df[cat_col].astype(str).value_counts().head(top_n)
        plot_hbar(counts, title=f"Top {top_n} — {cat_col}")
    else:
        st.info("Nenhuma coluna categórica detectada.")

# Histograma (numéricas)
with st.container():
    st.markdown("**Distribuição numérica (histograma)**")
    if num_cols:
        num_col = st.selectbox("Coluna numérica", num_cols, index=0, key="num_for_hist")
        bins = st.slider("Bins", min_value=5, max_value=100, value=30)
        plot_hist(df[num_col], title=f"Distribuição — {num_col}", bins=bins)
    else:
        st.info("Nenhuma coluna numérica detectada.")

# ------------------------------
# Mapa (se houver colunas lat/lon)
# ------------------------------
st.subheader("🗺️ Mapa (se houver lat/lon)")
cand_lat = [c for c in df.columns if c.lower() in ("lat", "latitude")]
cand_lon = [c for c in df.columns if c.lower() in ("lon", "long", "longitude")]

if cand_lat and cand_lon:
    lat_c = cand_lat[0]
    lon_c = cand_lon[0]
    st.map(df[[lat_c, lon_c]].dropna().rename(columns={lat_c: "lat", lon_c: "lon"}))
else:
    st.caption("Adicione colunas `lat`/`lon` (ou `latitude`/`longitude`) para habilitar o mapa.")

st.subheader("⬇️ Baixar dados filtrados")
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("Baixar CSV", data=csv_bytes, file_name="dados_filtrados.csv", mime="text/csv")

st.caption("💡 Dica: ajuste nomes de colunas/visualizações conforme a sua base.")
