import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from pathlib import Path  # <-- (ADAPT) para caminhos relativos

# =========================
# CONFIG & TEMA
# =========================
st.set_page_config(page_title="Análise frota CTCP - Pelotas (RS)", layout="wide")
st.title("Análise frota CTCP - Pelotas (RS)")

# =========================
# DADOS
# =========================
# (ADAPT) valor padrão agora é relativo ao repo; você ainda pode digitar outro caminho
csv_path = st.text_input(
    "Caminho do arquivo CSV",
    "dados_ctcp.csv"   # antes: r"C:\Users\israe\Desktop\analise tabela\dados_ctcp.csv"
)

# (ADAPT) procura o CSV de forma inteligente (repo raiz, subpasta data/, caminho digitado etc.)
APP_DIR = Path(__file__).parent
CANDIDATES = [
    # o que foi digitado, se existir como caminho absoluto/relativo
    Path(csv_path) if csv_path.strip() else None,
    APP_DIR / csv_path.strip() if csv_path.strip() else None,
    # defaults do repositório
    APP_DIR / "dados_ctcp.csv",
    APP_DIR / "data" / "dados_ctcp.csv",
]

def _resolve_csv():
    for c in CANDIDATES:
        try:
            if c and c.is_file():
                return c
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
def load_data(pathlike):
    # tenta utf-8 e faz fallback para latin-1
    try:
        return pd.read_csv(pathlike, encoding="utf-8")
    except Exception:
        return pd.read_csv(pathlike, encoding="latin-1")

csv_file = _resolve_csv()
if not csv_file:
    st.error(
        "CSV não encontrado. Coloque **dados_ctcp.csv** na raiz do repositório "
        "ou informe um caminho válido no campo acima (pode ser relativo, ex.: `data/dados_ctcp.csv`)."
    )
    st.stop()

try:
    df = load_data(csv_file)
except Exception as e:
    st.error(f"Erro ao ler o CSV: {e}")
    st.stop()

# Preparação
ano_atual = datetime.now().year
if "idade" not in df.columns:
    df["idade"] = ano_atual - df["Ano"]

# =========================
# SIDEBAR — FILTROS
# =========================
st.sidebar.header("Filtros")

ano_min, ano_max = int(df["Ano"].min()), int(df["Ano"].max())
f_ano = st.sidebar.slider("Ano (fabr.)", ano_min, ano_max, (ano_min, ano_max), step=1)

opts_emp = sorted(df["Empresa"].dropna().unique().tolist())
f_emp = st.sidebar.multiselect("Empresas", opts_emp, default=opts_emp)

opts_car = sorted(df["Carroceria"].dropna().unique().tolist())
f_car = st.sidebar.multiselect("Carrocerias", opts_car, default=opts_car)

opts_ch = sorted(df["Chassi"].dropna().unique().tolist())
f_ch = st.sidebar.multiselect("Chassis", opts_ch, default=opts_ch)

opts_let = sorted(df["Letreiro"].dropna().unique().tolist())
f_let = st.sidebar.multiselect("Letreiros", opts_let, default=opts_let)

opts_elev = df["Elevador"].dropna().unique().tolist()
f_elev = st.sidebar.multiselect("Elevador", opts_elev, default=opts_elev)

opts_ar = df["Ar"].dropna().unique().tolist()
f_ar = st.sidebar.multiselect("Ar-condicionado", opts_ar, default=opts_ar)

# Controle Top-N
TOP_N = st.sidebar.slider("Top-N (Carroceria/Chassi/Letreiro)", 5, 30, 12, step=1)

# Aplica filtros
mask = (
    df["Ano"].between(f_ano[0], f_ano[1]) &
    df["Empresa"].isin(f_emp) &
    df["Carroceria"].isin(f_car) &
    df["Chassi"].isin(f_ch) &
    df["Letreiro"].isin(f_let) &
    df["Elevador"].isin(f_elev) &
    df["Ar"].isin(f_ar)
)
df_f = df.loc[mask].copy()

# =========================
# MÉTRICAS 
# =========================
frota = int(df_f["Prefixo"].count())
idade_media = float(df_f["idade"].mean()) if len(df_f) else 0.0

st.markdown("""
<style>
.kpi-wrap {display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 14px; margin: 4px 0 8px 0;}
.kpi-card {
  background: #0f172a;
  color: #e5e7eb;
  border: 1px solid #1f2937;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 6px 18px rgba(0,0,0,.12);
}
.kpi-title {
  font-size: 0.95rem;
  letter-spacing: .02em;
  color: #cbd5e1;
  margin: 0 0 8px 0;
  font-weight: 600;
  text-transform: uppercase;
}
.kpi-value {
  font-size: 2.6rem;
  line-height: 1.1;
  font-weight: 800;
  color: #f8fafc;
  margin: 0 0 6px 0;
}
.kpi-sub {
  font-size: .95rem;
  color: #94a3b8;
  margin: 0;
}
.kpi-accent { color: #60a5fa; }
.kpi-accent-2 { color: #34d399; }
</style>

<div class="kpi-wrap">
  <div class="kpi-card">
    <div class="kpi-title">FROTA CTCP</div>
    <div class="kpi-value kpi-accent">""" + f"{frota:,}".replace(",", ".") + """</div>
    <p class="kpi-sub">Total de veículos atualmente no recorte selecionado</p>
  </div>
  <div class="kpi-card">
    <div class="kpi-title">IDADE MÉDIA DA FROTA</div>
    <div class="kpi-value kpi-accent-2">""" + f"{idade_media:.2f}".replace(".", ",") + """ <span style="font-size:1.4rem;font-weight:700;">anos</span></div>
    <p class="kpi-sub">Média de idade considerando “Ano” do veículo</p>
  </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# =========================
# HELPERS VISUAIS — CONFIG POR GRÁFICO
# =========================
FACE = "#FFFFFF"

# Configurações individuais de cada gráfico
CHART_CFG = {
    # Linha 1
    "mean_age_emp": {
        "figsize": (10, 10),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 10, "tick": 13, "title": 14}
        }
    },
    "frota_por_empresa": {
        "figsize": (10, 10),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 10, "tick": 13, "title": 16}
        }
    },
    # Linha 2
    "elevador": {
        "figsize": (3, 3),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 7, "tick": 7, "title": 6}
        }
    },
    "ar_condicionado": {
        "figsize": (3, 3),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 7, "tick": 7, "title": 6}
        }
    },
    # Linha 3
    "carrocerias_topN": {
        "figsize": (10, 10),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 10, "tick": 13, "title": 14}
        }
    },
    "chassis_topN": {
        "figsize": (10, 10),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 10, "tick": 13, "title": 14}
        }
    },
    # Linha 4 (full-width)
    "letreiros_topN": {
        "figsize": (18, 7.5),
        "font": {
            "family": "sans-serif",
            "name": "DejaVu Sans",
            "size": {"label": 10, "tick": 12, "title": 14}
        }
    },
}

def _rc_from_cfg(cfg):
    """Gera um dict de rcParams para família e nome de fonte."""
    fam = cfg["font"]["family"]
    name = cfg["font"]["name"]
    # Nota: para 'sans-serif', 'serif', etc., definimos a família e priorizamos o 'name'
    rc = {
        "font.family": fam,
        # Para cada família, apontamos a lista iniciando pelo nome desejado
        "font.sans-serif": [name],
        "font.serif": [name],
        "font.monospace": [name],
    }
    return rc

def hbar(series, title, cfg, xlabel="Quantidade", annotate_fmt="{:,.0f}"):
    """Barras horizontais ordenadas (para categorias) com config específica."""
    if series.empty:
        st.info("Sem dados para os filtros selecionados.")
        return
    series = series.sort_values(ascending=True)
    with plt.rc_context(_rc_from_cfg(cfg)):
        fig, ax = plt.subplots(figsize=cfg["figsize"])
        fig.set_facecolor(FACE)
        ax.barh(series.index, series.values, color=plt.cm.tab10.colors[:len(series)] )
        for i, v in enumerate(series.values):
            ax.text(
                v, i, "  " + annotate_fmt.format(v),
                va="center",
                fontsize=cfg["font"]["size"]["label"],
                weight="bold"
            )
        ax.set_xlabel(xlabel, fontsize=cfg["font"]["size"]["label"])
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=cfg["font"]["size"]["tick"])
        ax.set_title(title, pad=10, weight="bold", fontsize=cfg["font"]["size"]["title"])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

def bar_mean(series, title, cfg, ylabel="Idade média (anos)"):
    """Barras verticais para médias por grupo com config específica."""
    if series.empty:
        st.info("Sem dados para os filtros selecionados.")
        return
    series = series.sort_values(ascending=False)
    with plt.rc_context(_rc_from_cfg(cfg)):
        fig, ax = plt.subplots(figsize=cfg["figsize"])
        fig.set_facecolor(FACE)
        bars = ax.bar(series.index, series.values, color=plt.cm.tab10.colors[:len(series)] )
        for b in bars:
            ax.text(
                b.get_x()+b.get_width()/2, b.get_height()+0.06,
                f"{b.get_height():.2f}",
                ha="center", va="bottom",
                fontsize=cfg["font"]["size"]["label"],
                weight="bold"
            )
        ax.set_ylabel(ylabel, fontsize=cfg["font"]["size"]["label"])
        ax.set_xlabel("")
        plt.xticks(rotation=45, ha="right", fontsize=cfg["font"]["size"]["tick"])
        ax.tick_params(axis='y', labelsize=cfg["font"]["size"]["tick"])
        ax.set_title(title, pad=10, weight="bold", fontsize=cfg["font"]["size"]["title"])
        st.pyplot(fig, use_container_width=True)

def pie_chart(series, title, cfg):
    """Pizza para proporções (Elevador/Ar) com config específica."""
    if series.empty:
        st.info("Sem dados para os filtros selecionados.")
        return
    vals = series.values
    labs = series.index.tolist()
    cores = list(plt.cm.tab10.colors[:len(vals)])
    with plt.rc_context(_rc_from_cfg(cfg)):
        fig, ax = plt.subplots(figsize=cfg["figsize"])
        fig.set_facecolor(FACE)
        ax.pie(
            vals, labels=labs, autopct='%1.1f%%', startangle=90,
            colors=cores, wedgeprops=dict(edgecolor='white', linewidth=2),
            textprops=dict(
                color='black',
                weight='bold',
                fontsize=cfg["font"]["size"]["label"],
                family=cfg["font"]["family"],
                fontname=cfg["font"]["name"]
            )
        )
        ax.set_aspect('equal')
        plt.tight_layout()
        ax.set_title(title, pad=10, weight="bold", fontsize=cfg["font"]["size"]["title"])
        st.pyplot(fig, use_container_width=True)

# =========================
# LAYOUT EM 2 COLUNAS (maiores)
# =========================

# Linha 1 — 2 gráficos grandes
c1, c2 = st.columns(2, gap="large")
with c1:
    mean_age_emp = df_f.groupby("Empresa")["idade"].mean()
    bar_mean(mean_age_emp, "Idade média por empresa", cfg=CHART_CFG["mean_age_emp"])
with c2:
    emp_counts = df_f["Empresa"].value_counts()
    hbar(emp_counts, "Frota por empresa", cfg=CHART_CFG["frota_por_empresa"])

# Linha 2 — 2 gráficos (pizza maiores)
c3, c4 = st.columns(2, gap="large")
with c3:
    elev_counts = df_f["Elevador"].value_counts()
    pie_chart(elev_counts, "Elevador", cfg=CHART_CFG["elevador"])
with c4:
    ar_counts = df_f["Ar"].value_counts()
    pie_chart(ar_counts, "Ar condicionado", cfg=CHART_CFG["ar_condicionado"])

# Linha 3 — 2 gráficos de barras grandes
c5, c6 = st.columns(2, gap="large")
with c5:
    car_counts = df_f["Carroceria"].value_counts().head(TOP_N)
    hbar(car_counts, "Carrocerias", cfg=CHART_CFG["carrocerias_topN"])
with c6:
    ch_counts = df_f["Chassi"].value_counts().head(TOP_N)
    hbar(ch_counts, "Chassis", cfg=CHART_CFG["chassis_topN"])

# Linha 4 — gráfico full-width (Letreiros)
st.subheader("Letreiros")
let_counts = df_f["Letreiro"].value_counts().head(TOP_N)
if let_counts.empty:
    st.info("Sem dados para os filtros selecionados.")
else:
    hbar(
        let_counts,
        title="Letreiros",
        cfg=CHART_CFG["letreiros_topN"],
        xlabel="Quantidade",
        annotate_fmt="{:,.0f}"
    )

st.divider()

# =========================
# TABELA (após filtros)
# =========================
st.subheader("Tabela")
st.dataframe(df_f, use_container_width=True)
