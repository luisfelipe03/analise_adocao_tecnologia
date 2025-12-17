import sys
from pathlib import Path
import streamlit as st
import pandas as pd

FILE_PATH = Path(__file__).resolve()
INTERFACE_DIR = FILE_PATH.parent 
SRC_DIR = INTERFACE_DIR.parent 

sys.path.append(str(SRC_DIR))

try:
    from analysis.stats import *
except ImportError as e:
    st.error(f"Erro ao importar módulos. Verifique a estrutura de pastas. Detalhe: {e}")
    st.stop()

st.set_page_config(page_title="Adoção de Tecnologias", layout="wide")

@st.cache_data
def load_data():
    csv_path = SRC_DIR / "data" / "database.csv"
    
    try:
        return pd.read_csv(csv_path, sep=";", decimal=",")
    except FileNotFoundError:
        st.error(f"Arquivo não encontrado no caminho: {csv_path}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning("A base de dados está vazia ou não foi carregada corretamente.")
    st.stop()

ordem_periodos = ['Q1_2023', 'Q2_2023', 'Q3_2023', 'Q4_2023', 
                  'Q1_2024', 'Q2_2024', 'Q3_2024', 'Q4_2024', 'Q1_2025']
ordem_periodos = [p for p in ordem_periodos if p in df['Periodo'].unique()]
df['Periodo'] = pd.Categorical(df['Periodo'], categories=ordem_periodos, ordered=True)
df = df.sort_values('Periodo')

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

st.title("📊 Análise Estatística: Adoção de Tecnologias (2023-2025)")
st.markdown("*Dashboard interativo para suporte à apresentação de Estatística e Probabilidade.*")

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Adoção Média", f"{df_filtro['Taxa_Adocao_Percent'].mean():.2f}%")
kpi2.metric("Investimento Médio", f"R$ {df_filtro['Investimento_Milhoes'].mean():.2f} Mi")
kpi3.metric("Satisfação Média", f"{df_filtro['Satisfacao_Media'].mean():.2f}")
kpi4.metric("Tempo Implementação", f"{df_filtro['Tempo_Implementacao_Meses'].mean():.2f} Meses")

st.subheader("Classificação das Variáveis")
st.markdown("""
Para esta análise estatística, classificamos as variáveis da seguinte forma:

| Natureza | Tipo | Variáveis do Estudo |
| :--- | :--- | :--- |
| **Qualitativa** | *Nominal* | `Tecnologia` |
| **Qualitativa** | *Ordinal* | `Periodo` (Ordem Cronológica) |
| **Quantitativa** | *Discreta* | `Empresas_Adotantes`, `Profissionais_Treinados` |
| **Quantitativa** | *Contínua* | `Taxa_Adocao`, `Investimento`, `Satisfacao`, `Tempo_Implementacao` |
""")

st.markdown("---")

st.header("1. Visão Geral da Amostra")
st.caption("Primeiras linhas da base de dados carregada.")
st.dataframe(df_filtro.head(), use_container_width=True)

st.info("**Nota Metodológica:** A variável resposta ($Y$) deste estudo é a **Taxa de Adoção**.")

st.header("2. Estatística Descritiva")
st.markdown("Observe o **CV (%)** para analisar volatilidade e a **Distorção** para assimetria.")
st.dataframe(estatistica_descritiva(df_filtro), use_container_width=True)

col_desc1, col_desc2 = st.columns(2)
with col_desc1:
    st.pyplot(histograma_adocao(df_filtro))
with col_desc2:
    st.info("""
    **Pontos de Atenção:**
    - O **Investimento** possui alta variabilidade (CV alto).
    - A **Satisfação** é consistente (CV baixo).
    """)

st.markdown("---")
st.header("3. Comparação e Eficiência")

col_comp1, col_comp2 = st.columns(2)
with col_comp1:
    st.pyplot(ranking_medio_adocao(df_filtro))
with col_comp2:
    st.pyplot(boxplot_tempo(df_filtro))

st.markdown("> **Insight:** API REST tem alta adoção com o menor tempo de implementação (mediana baixa).")

st.markdown("---")
st.header("4. Evolução Temporal")
st.markdown("Comparativo de crescimento entre todas as tecnologias selecionadas.")

st.pyplot(grafico_evolucao_comparativo(df_filtro))

st.markdown("---")
st.header("5. Correlações e Fatores")

col_corr1, col_corr2 = st.columns(2)
with col_corr1:
    st.pyplot(matriz_correlacao(df_filtro))
with col_corr2:
    st.pyplot(dispersao_satisfacao(df_filtro))

st.markdown("> **Insight:** A correlação entre Satisfação e Adoção é fortíssima (próxima de 1.0).")

st.markdown("---")
st.header("6. Probabilidade e Conclusão")

p_simples, p_condicional = calcular_probabilidades(df_filtro)
aumento_percentual = (p_condicional - p_simples) * 100

st.subheader("🎲 Previsão de Cenários (Frequentista)")

c_prob1, c_prob2, c_prob3 = st.columns([1, 1, 1.5])

with c_prob1:
    st.metric(
        label="Cenário Base (Aleatório)", 
        value=f"{p_simples:.1%}",
        help="Probabilidade de uma tecnologia qualquer ter Alta Adoção (>40%)"
    )
    st.caption("Chance de sucesso sem estratégia definida.")

with c_prob2:
    st.metric(
        label="Cenário Alta Investimento", 
        value=f"{p_condicional:.1%}",
        delta=f"+{aumento_percentual:.1f} p.p.",
        help="Probabilidade dado que o Investimento > Média"
    )
    st.caption("O investimento pesado quase **dobra** a chance de sucesso.")

with c_prob3:
    df_prob = pd.DataFrame({
        "Cenário": ["Base", "Alto Invest."],
        "Probabilidade": [p_simples, p_condicional]
    })
    fig_prob, ax_prob = plt.subplots(figsize=(4, 2))
    sns.barplot(data=df_prob, x="Probabilidade", y="Cenário", ax=ax_prob, palette=["gray", "#2ecc71"])
    ax_prob.set_xlim(0, 1)
    ax_prob.set_xlabel("Probabilidade de Sucesso")
    ax_prob.set_ylabel("")
    for i, v in enumerate([p_simples, p_condicional]):
        ax_prob.text(v + 0.02, i, f"{v:.1%}", va='center', fontweight='bold')
    st.pyplot(fig_prob)

st.markdown("---")
st.subheader("🏆 Conclusões Finais: Os 3 Pilares da Adoção")

col_conc1, col_conc2, col_conc3 = st.columns(3)

with col_conc1:
    st.info("**1. Agilidade (Time-to-Market)**")
    st.markdown("""
    Tecnologias de implementação rápida, como **API REST**, saem na frente.
    Menor tempo reduz barreiras de entrada.
    """)

with col_conc2:
    st.success("**2. Qualidade (Experiência)**")
    st.markdown("""
    A **Satisfação** é o fiel da balança.
    Com correlação de **0,95**, focar na experiência do usuário garante a retenção.
    """)

with col_conc3:
    st.warning("**3. Tendência (O Futuro)**")
    st.markdown("""
    **Machine Learning** apresentou crescimento exponencial.
    Apesar do custo, é onde reside a inovação estratégica a longo prazo.
    """)