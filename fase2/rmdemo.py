import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

from regmultipla import regmultipla

# Estilo visual para os gráficos
plt.style.use("seaborn-v0_8-whitegrid")


def main():
    # a) Faça o download dos dados do arquivo data.mat ou data.csv. A primeira coluna é o tamanho da casa, a segunda coluna é o número de quartos, e a terceira coluna é o preço da casa.
    # Tentamos localizar o arquivo data.csv na pasta fase2 ou na raiz
    csv_path = Path("fase2/data.csv")
    if not csv_path.exists():
        csv_path = Path("data.csv")
        
    dados_df = pd.read_csv(csv_path, header=None, names=["tamanho", "quartos", "preco"])
    print("a) Dados carregados com sucesso.")

    # b) Utilize o comando python .describe() para fazer uma primeira análise estatística da sua base de dados. Qual a média de preço das casas? Quanto custa a menor casa? Quantos quartos tem a casa mais cara?
    print("\nb) Análise estatística (.describe()):")
    print(dados_df.describe())
    
    media_preco = dados_df["preco"].mean()
    preco_menor_casa = dados_df.loc[dados_df["tamanho"].idxmin(), "preco"]
    quartos_mais_cara = int(dados_df.loc[dados_df["preco"].idxmax(), "quartos"])
    
    print(f"Média de preço: {media_preco:.2f}")
    print(f"Preço da menor casa (pelo tamanho): {preco_menor_casa:.2f}")
    print(f"Quartos da casa mais cara: {quartos_mais_cara}")

    # c) Gere uma matriz X para as variáveis independentes (que são o tamanho da casa e o número de quartos) e o vetor da variável dependente (que é o preço).
    X_base = dados_df[["tamanho", "quartos"]].to_numpy()
    y = dados_df["preco"].to_numpy().reshape(-1, 1)
    
    # Adicionando a coluna de 1s para o intercepto (conforme figura da Matriz X no enunciado)
    m = X_base.shape[0]
    X = np.column_stack([np.ones(m), X_base])
    print("\nc) Matriz X e vetor y gerados com sucesso.")

    # d) Verifique a correlação e a regressão para Tamanho da casa e Preço, e, Número de quartos e Preço e apresente os valores no gráfico de dispersão.
    tamanho = X_base[:, 0]
    quartos = X_base[:, 1]
    y_plot = y.flatten()

    r_tam = np.corrcoef(tamanho, y_plot)[0, 1]
    r_qua = np.corrcoef(quartos, y_plot)[0, 1]
    
    # Cálculo das retas individuais
    b1_tam, b0_tam = np.polyfit(tamanho, y_plot, 1)
    b1_qua, b0_qua = np.polyfit(quartos, y_plot, 1)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(tamanho, y_plot, edgecolors="blue", facecolors="none")
    plt.plot(tamanho, b0_tam + b1_tam * tamanho, "r-")
    plt.title(f"d) Tamanho x Preço | r={r_tam:.4f}")
    
    plt.subplot(1, 2, 2)
    plt.scatter(quartos, y_plot, edgecolors="blue", facecolors="none")
    plt.plot(quartos, b0_qua + b1_qua * quartos, "r-")
    plt.title(f"d) Quartos x Preço | r={r_qua:.4f}")

    # e) Calcule a Regressão Múltipla. Faça o gráfico de dispersão em 3D com o tamanho da casa, número de quartos, e o preço da casa. Neste caso iremos trabalhar com o espaço 3D (verifique como usar Axes3D).
    theta = regmultipla(X, y)
    print("\ne) Regressão Múltipla calculada.")
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tamanho, quartos, y_plot, edgecolors="blue", facecolors="none", s=50)
    
    # f) Trace a linha da regressão no Gráfico de Dispersão. Você pode girar este gráfico para visualizar melhor os dados.
    y_reg = theta[0] + theta[1] * tamanho + theta[2] * quartos
    # Representação visual da linha de regressão múltipla conectando os pontos
    ax.plot(tamanho, quartos, y_reg.flatten(), "r-", linewidth=2, label="Previsão do Modelo")
    
    ax.set_xlabel("Tamanho")
    ax.set_ylabel("Quartos")
    ax.set_zlabel("Preço")

    # ========================
    # GRAFICO 2
    # ========================
    plt.figure(figsize=(6, 4))

    plt.scatter(
        quartos,
        y,
        facecolors="none",
        edgecolors="blue",
        s=40
    )

    x_linha2 = np.linspace(quartos.min(), quartos.max(), 100)
    plt.plot(x_linha2, b0_qua + b1_qua * x_linha2, "r-", linewidth=1.5)

    plt.title(f"Coeficiente: {r_qua:.4f}", fontsize=10, fontweight="bold")
    plt.xlabel("Numero de Quartos", fontsize=9)
    plt.ylabel("Preco por R$ 10.000", fontsize=9)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(alpha=0.3)

    # ========================
    # GRAFICO 3D (PADRAO PROFESSORA)
    # ========================
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        tamanho,
        quartos,
        y,
        facecolors="none",
        edgecolors="blue",
        s=40
    )

    # regressao multipla
    y_reg = theta[0, 0] + theta[1, 0] * tamanho + theta[2, 0] * quartos

    # linha quebrada estilo da professora
    for i in range(len(tamanho) - 1):
        ax.plot(
            tamanho[i:i+2],
            quartos[i:i+2],
            y_reg[i:i+2],
            "r-",
            linewidth=1
        )

    ax.set_xlabel("Tamanho da Casa", fontsize=9, labelpad=10)
    ax.set_ylabel("Numero de Quartos", fontsize=9, labelpad=10)
    ax.set_zlabel("Preco por R$ 10.000", fontsize=9, labelpad=10)

    ax.set_title(
        "Coeficientes: %.2f %.2f %.2f"
        % (theta[0, 0], theta[1, 0], theta[2, 0]),
        fontsize=10,
        fontweight="bold",
        pad=15
    )

    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.view_init(elev=25, azim=-60)

    # previsao
    x_novo = np.array([1, 1650, 3], dtype=float)
    preco_previsto = float(x_novo @ theta[:, 0])
    print(f"\nPreco previsto para casa de 1650 pes e 3 quartos: {preco_previsto:.2f}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
