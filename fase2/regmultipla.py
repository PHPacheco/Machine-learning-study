import numpy as np


# Semelhante a fase anterior, você deve implementar a função regmultipla que calcula os parâmetros beta para os dados de entrada y e X.
def regmultipla(X, y):
    # Converte para array numpy para garantir compatibilidade
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo numero de linhas.")

    # Resolução da Equação Normal conforme o enunciado: beta = (X' X)^-1 X' y
    # O comando np.linalg.solve é a forma mais estável de resolver o sistema (X'X) * beta = X'y
    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return theta
