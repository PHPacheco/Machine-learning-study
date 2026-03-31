import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

from regmultipla import regmultipla

# Estilo visual para os gráficos
plt.style.use("seaborn-v0_8-whitegrid")

def main():
    # a) Faça o download dos dados do arquivo data.mat ou data.csv. 
    # A primeira coluna é o tamanho da casa, a segunda coluna é o número de quartos, e a terceira coluna é o preço da casa.
    try:
        # Tenta carregar o arquivo .mat primeiro, caso exista na pasta fase2
        if Path("fase2/data.mat").exists():
            mat = loadmat("fase2/data.mat")
            data = mat['data']
            dados_df = pd.DataFrame(data, columns=["tamanho", "quartos", "preco"])
        # Caso contrário, tenta carregar o arquivo .csv
        elif Path("fase2/data.csv").exists():
            dados_df = pd.read_csv("fase2/data.csv", header=None, names=["tamanho", "quartos", "preco"])
        else:
            raise FileNotFoundError("Nenhum arquivo data.csv ou data.mat encontrado na pasta fase2.")
            
        print("a) Dados carregados com sucesso.")
    except Exception as e:
        print(f"Erro ao carregar os dados: {e}")
        return

    # b) Utilize o comando python .describe() para fazer uma primeira análise estatística da sua base de dados. 
    # Qual a média de preço das casas? Quanto custa a menor casa? Quantos quartos tem a casa mais cara?
    print("\nb) Análise estatística (.describe()):")
    print(dados_df.describe())
    
    media_preco = dados_df["preco"].mean()
    preco_menor_casa = dados_df.loc[dados_df["tamanho"].idxmin(), "preco"]
    quartos_mais_cara = int(dados_df.loc[dados_df["preco"].idxmax(), "quartos"])
    
    print(f"Respostas do item (b):")
    print(f"- Média de preço das casas: {media_preco:.2f}")
    print(f"- Preço da menor casa (pelo tamanho): {preco_menor_casa:.2f}")
    print(f"- Quartos da casa mais cara: {quartos_mais_cara}")

    # c) Gere uma matriz X para as variáveis independentes (que são o tamanho da casa e o número de quartos) 
    # e o vetor da variável dependente (que é o preço).
    X_base = dados_df[["tamanho", "quartos"]].to_numpy()
    y = dados_df["preco"].to_numpy().reshape(-1, 1)
    
    # Adicionando a coluna de 1s para o intercepto conforme a figura da Matriz X no enunciado
    m = X_base.shape[0]
    X = np.column_stack([np.ones(m), X_base])
    print("\nc) Matriz X (com coluna de 1s) e vetor y gerados.")

    # d) Verifique a correlação e a regressão para Tamanho da casa e Preço, e, Número de quartos e Preço e apresente os valores no gráfico de dispersão.
    tamanho = X_base[:, 0]
    quartos = X_base[:, 1]
    y_plot = y.flatten()

    r_tam = np.corrcoef(tamanho, y_plot)[0, 1]
    r_qua = np.corrcoef(quartos, y_plot)[0, 1]
    
    # Cálculo das retas individuais para visualização
    b1_tam, b0_tam = np.polyfit(tamanho, y_plot, 1)
    b1_qua, b0_qua = np.polyfit(quartos, y_plot, 1)

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.scatter(tamanho, y_plot, edgecolors="blue", facecolors="none")
    plt.plot(tamanho, b0_tam + b1_tam * tamanho, "r-")
    plt.title(f"Tamanho da casa e Preço\nCorrelação: {r_tam:.4f}")
    
    plt.subplot(1, 2, 2)
    plt.scatter(quartos, y_plot, edgecolors="blue", facecolors="none")
    plt.plot(quartos, b0_qua + b1_qua * quartos, "r-")
    plt.title(f"Número de quartos e Preço\nCorrelação: {r_qua:.4f}")

    # e) Calcule a Regressão Múltipla. Faça o gráfico de dispersão em 3D com o tamanho da casa, 
    # número de quartos, e o preço da casa. Neste caso iremos trabalhar com o espaço 3D (verifique como usar Axes3D).
    theta = regmultipla(X, y)
    print("\ne) Regressão Múltipla calculada.")
    
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(tamanho, quartos, y_plot, edgecolors="blue", facecolors="none", s=50)

if __name__ == "__main__":
    main()
