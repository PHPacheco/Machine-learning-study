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
    
    # f) Trace a linha da regressão no Gráfico de Dispersão. Você pode girar este gráfico para visualizar melhor os dados.
    # Usamos theta.flatten() para garantir que os coeficientes sejam tratados como escalares no cálculo
    t0, t1, t2 = theta.flatten()
    y_reg = t0 + t1 * tamanho + t2 * quartos
    ax.plot(tamanho, quartos, y_reg, "r-", linewidth=2)
    
    ax.set_xlabel("Tamanho da casa")
    ax.set_ylabel("Número de quartos")
    ax.set_zlabel("Preço da casa")

    # g) Mostre na figura os coeficientes de correlação entre Tamanho da casa e Preço e Número de quartos e Preço.
    ax.set_title(f"Correlação: Tamanho da casa e Preço ({r_tam:.4f})\nCorrelação: Número de quartos e Preço ({r_qua:.4f})")

    # h) Calcule o preço de uma casa que tem tamanho de 1650 e 3 quartos. O resultado deve ser igual a 293081. 
    # Aumente e diminua a quantidade de número de quartos. O que acontece? Por qual motivo?
    # O uso do .item() extrai o valor numérico de dentro do array resultante
    preco_previsto_3q = (np.array([1, 1650, 3]) @ theta).item()
    preco_previsto_2q = (np.array([1, 1650, 2]) @ theta).item()
    preco_previsto_4q = (np.array([1, 1650, 4]) @ theta).item()
    
    print(f"\nh) Teste de previsão (1650 pés, 3 quartos): {preco_previsto_3q:.2f}")
    print(f"   (Valor esperado no enunciado: 293081)")
    print(f"Preço com 2 quartos: {preco_previsto_2q:.2f}")
    print(f"Preço com 4 quartos: {preco_previsto_4q:.2f}")
    print("Motivo: O preço aumenta com mais quartos pois o coeficiente correspondente (theta2) é positivo.")

    # i) Compare seus resultados com a função de regressão linear múltipla do python. 
    # Para isso você irá precisar das bibliotecas numpy e scikit-learn.
    model_sk = LinearRegression()
    model_sk.fit(X_base, y)
    theta_sk = np.array([model_sk.intercept_[0], model_sk.coef_[0][0], model_sk.coef_[0][1]])
    
    print("\ni) Comparação com Scikit-Learn:")
    print(f"- Coeficientes (Equação Normal): {theta.flatten()}")
    print(f"- Coeficientes (Scikit-Learn): {theta_sk}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
