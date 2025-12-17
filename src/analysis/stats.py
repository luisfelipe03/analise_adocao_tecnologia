import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

NUMERIC_COLS = [
    "Empresas_Adotantes",
    "Taxa_Adocao_Percent",
    "Investimento_Milhoes",
    "Profissionais_Treinados",
    "Satisfacao_Media",
    "Tempo_Implementacao_Meses",
]

def estatistica_descritiva(df):

    if df.empty:
        return pd.DataFrame()
        
    desc = df[NUMERIC_COLS].describe().T
    desc["mediana"] = df[NUMERIC_COLS].median()
    desc["CV (%)"] = (df[NUMERIC_COLS].std() / df[NUMERIC_COLS].mean()) * 100
    desc["skewness"] = df[NUMERIC_COLS].skew()
    
    cols_final = ["count", "mean", "mediana", "std", "min", "max", "CV (%)", "skewness"]
    return desc[cols_final].round(2)

def grafico_evolucao_comparativo(df):
    """Gráfico de linhas com todas as tecnologias juntas."""
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(
        data=df, 
        x="Periodo", 
        y="Taxa_Adocao_Percent", 
        hue="Tecnologia", 
        marker="o", 
        ax=ax,
        palette="tab10"
    )
    ax.set_title("Evolução Comparativa da Adoção (2023-2025)")
    ax.set_ylabel("Taxa de Adoção (%)")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=45)
    return fig

def histograma_adocao(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["Taxa_Adocao_Percent"], kde=True, ax=ax, color="#3498db")
    ax.set_title("Histograma: Distribuição da Taxa de Adoção")
    ax.set_xlabel("Taxa (%)")
    return fig

def ranking_medio_adocao(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    rank = df.groupby("Tecnologia")["Taxa_Adocao_Percent"].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=rank, x="Taxa_Adocao_Percent", y="Tecnologia", ax=ax, palette="viridis")
    ax.set_title("Ranking Médio de Adoção")
    ax.set_xlabel("Média (%)")
    return fig

def boxplot_tempo(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="Tecnologia", y="Tempo_Implementacao_Meses", ax=ax, palette="Set2")
    ax.set_title("Boxplot: Tempo de Implementação")
    ax.set_ylabel("Meses")
    plt.xticks(rotation=45)
    return fig

def dispersao_satisfacao(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=df, 
        x="Satisfacao_Media", 
        y="Taxa_Adocao_Percent", 
        hue="Tecnologia", 
        s=100, 
        ax=ax
    )
    ax.set_title("Correlação: Satisfação vs. Adoção")
    ax.set_xlabel("Satisfação Média (Nota)")
    ax.set_ylabel("Adoção (%)")
    ax.grid(True)
    return fig

def matriz_correlacao(df):
    fig, ax = plt.subplots(figsize=(6, 5))
    corr = df[NUMERIC_COLS].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title("Matriz de Correlação de Pearson")
    return fig

def calcular_probabilidades(df):
    total_obs = len(df)
    if total_obs == 0:
        return 0.0, 0.0

    # Probabilidade Simples: P(Adoção > 40%)
    alta_adocao = df[df['Taxa_Adocao_Percent'] > 40]
    p_simples = len(alta_adocao) / total_obs
    
    # Probabilidade Condicional: P(Adoção > 40% | Investimento > Média)
    media_inv = df['Investimento_Milhoes'].mean()
    alto_investimento = df[df['Investimento_Milhoes'] > media_inv]
    
    if len(alto_investimento) > 0:
        sucesso_com_investimento = alto_investimento[alto_investimento['Taxa_Adocao_Percent'] > 40]
        p_condicional = len(sucesso_com_investimento) / len(alto_investimento)
    else:
        p_condicional = 0.0
        
    return p_simples, p_condicional

def grafico_eficiencia_custo(df):
    """
    Gráfico de Custo x Benefício: Quanto custa cada 1% de Market Share?
    Fundamental para o argumento do Integrante 3.
    """
    # 1. Agrupa e calcula as médias
    metrics = df.groupby('Tecnologia')[['Taxa_Adocao_Percent', 'Investimento_Milhoes']].mean()
    
    # 2. Calcula o indicador: Investimento Necessário para 1% de Adoção
    # (Quanto menor, mais eficiente)
    metrics['Custo_por_Ponto'] = metrics['Investimento_Milhoes'] / metrics['Taxa_Adocao_Percent']
    metrics = metrics.sort_values('Custo_por_Ponto', ascending=True).reset_index()

    # 3. Plotagem
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Paleta customizada: Verdes para os eficientes, Vermelhos para os caros
    # Usando 'RdYlGn_r' (Red-Yellow-Green reversed) para que valores baixos sejam verdes
    sns.barplot(
        data=metrics, 
        x='Custo_por_Ponto', 
        y='Tecnologia', 
        palette='RdYlGn_r', 
        ax=ax
    )
    
    # Títulos e Eixos
    ax.set_title("Custo Médio para Conquistar 1% de Mercado (Menor é Melhor)")
    ax.set_xlabel("Investimento (Milhões) por 1% de Adoção")
    ax.set_ylabel("")
    
    # Adiciona os valores nas barras para facilitar a leitura
    for i, v in enumerate(metrics['Custo_por_Ponto']):
        ax.text(v + 0.02, i, f"R$ {v:.2f} Mi", va='center', fontweight='bold', fontsize=10)
        
    # Linha de referência do Cloud (Benchmark)
    # Pega o valor do Cloud para desenhar a linha
    val_cloud = metrics.loc[metrics['Tecnologia'] == 'Cloud_Computing', 'Custo_por_Ponto'].values[0]
    ax.axvline(val_cloud, color='red', linestyle='--', alpha=0.5)
    ax.text(val_cloud, len(metrics)-0.5, " Benchmark (Cloud)", color='red', va='center')
    
    return fig