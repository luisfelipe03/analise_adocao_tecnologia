import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.analysis.stats import *

st.set_page_config(page_title="Adoção de Tecnologias", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv(
        "src/data/database.csv",
        sep=";",
        decimal=","
    )

df = load_data()

# Filtros
st.sidebar.title("Filtros")

periodos = st.sidebar.multiselect(
    "Períodos",
    options=df["Periodo"].unique(),
    default=df["Periodo"].unique()
)

tecnologias = st.sidebar.multiselect(
    "Tecnologias",
    options=df["Tecnologia"].unique(),
    default=df["Tecnologia"].unique()
)

df_filtro = df[
    df["Periodo"].isin(periodos) &
    df["Tecnologia"].isin(tecnologias)
]

st.title("Dashboard Analítico – Adoção de Tecnologias")

# Métricas principais
c1, c2, c3, c4 = st.columns(4)

c1.metric("Taxa Média (%)", round(df_filtro["Taxa_Adocao_Percent"].mean(), 2))
c2.metric("Investimento Médio (Mi)", round(df_filtro["Investimento_Milhoes"].mean(), 2))
c3.metric("Satisfação Média", round(df_filtro["Satisfacao_Media"].mean(), 2))
c4.metric("Tempo Médio (meses)", round(df_filtro["Tempo_Implementacao_Meses"].mean(), 2))

# Estatística Descritiva
st.header("Estatística Descritiva")
st.dataframe(estatistica_descritiva(df_filtro), use_container_width=True)

# Evolução Temporal
st.header("Evolução Temporal")

tech = st.selectbox("Tecnologia", tecnologias)
df_time = df_filtro[df_filtro["Tecnologia"] == tech]

fig_evolucao = grafico_evolucao(df_time, tech)
fig_evolucao.set_size_inches(6, 3)
st.pyplot(fig_evolucao)

# Distribuição
st.header("Distribuição da Taxa de Adoção")

col1, col2 = st.columns(2)

with col1:
    fig_hist = histograma_adocao(df_filtro)
    fig_hist.set_size_inches(5, 3)
    st.pyplot(fig_hist)

with col2:
    fig_box = boxplot_adocao(df_filtro)
    fig_box.set_size_inches(5, 3)
    st.pyplot(fig_box)

# Comparações
st.header("Comparações entre Tecnologias")

metrica = st.selectbox("Métrica de comparação", NUMERIC_COLS)

fig_rank = ranking_medio(df_filtro, metrica)
fig_rank.set_size_inches(6, 3)
st.pyplot(fig_rank)

# Relações
st.header("Relações entre Variáveis")

col3, col4 = st.columns(2)

with col3:
    fig_disp = dispersao_investimento(df_filtro)
    fig_disp.set_size_inches(5, 3)
    st.pyplot(fig_disp)

with col4:
    fig_corr = matriz_correlacao(df_filtro)
    fig_corr.set_size_inches(5, 4)
    st.pyplot(fig_corr)

# Conclusões
st.header("Conclusões")
st.markdown("""
- A adoção das tecnologias apresenta crescimento consistente ao longo do tempo.
- Tecnologias com maior investimento e maior número de profissionais treinados tendem a apresentar maiores taxas de adoção.
- O tempo médio de implementação diminui à medida que a tecnologia amadurece.
- Cloud Computing e API REST destacam-se como líderes de mercado no período analisado.
""")
