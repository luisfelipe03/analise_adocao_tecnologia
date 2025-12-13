import matplotlib.pyplot as plt

NUMERIC_COLS = [
    "Empresas_Adotantes",
    "Taxa_Adocao_Percent",
    "Investimento_Milhoes",
    "Profissionais_Treinados",
    "Satisfacao_Media",
    "Tempo_Implementacao_Meses",
]

def estatistica_descritiva(df):
    desc = df[NUMERIC_COLS].describe().T
    desc["mediana"] = df[NUMERIC_COLS].median()
    desc["variancia"] = df[NUMERIC_COLS].var()
    desc["coef_var_%"] = (desc["std"] / desc["mean"]) * 100
    return desc.round(2)

def grafico_evolucao(df, tecnologia):
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.plot(df["Periodo"], df["Taxa_Adocao_Percent"], marker="o")
    ax.set_title(f"Evolução da Adoção – {tecnologia}")
    ax.set_xlabel("Período")
    ax.set_ylabel("Taxa (%)")
    plt.xticks(rotation=45)
    return fig

def histograma_adocao(df):
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.hist(df["Taxa_Adocao_Percent"], bins=10)
    ax.set_title("Distribuição da Taxa de Adoção")
    return fig

def boxplot_adocao(df):
    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    df.boxplot(column="Taxa_Adocao_Percent", by="Tecnologia", ax=ax)
    plt.suptitle("")
    plt.xticks(rotation=45)
    return fig

def ranking_medio(df, coluna):
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    df.groupby("Tecnologia")[coluna].mean().sort_values().plot(kind="barh", ax=ax)
    ax.set_title(f"Ranking Médio – {coluna}")
    return fig

def dispersao_investimento(df):
    fig, ax = plt.subplots(figsize=(4.5, 2.5))
    ax.scatter(df["Investimento_Milhoes"], df["Taxa_Adocao_Percent"])
    ax.set_xlabel("Investimento (Mi)")
    ax.set_ylabel("Taxa de Adoção (%)")
    return fig

def matriz_correlacao(df):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    corr = df[NUMERIC_COLS].corr()
    im = ax.imshow(corr)
    ax.set_xticks(range(len(NUMERIC_COLS)))
    ax.set_yticks(range(len(NUMERIC_COLS)))
    ax.set_xticklabels(NUMERIC_COLS, rotation=45, ha="right")
    ax.set_yticklabels(NUMERIC_COLS)
    fig.colorbar(im)
    ax.set_title("Matriz de Correlação")
    return fig
