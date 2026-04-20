import numpy as np

def regmultipla(X, y):
    # Converte para array numpy para garantir compatibilidade
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)

    if X.shape[0] != y.shape[0]:
        raise ValueError("X e y devem ter o mesmo numero de linhas.")

    theta = np.linalg.solve(X.T @ X, X.T @ y)
    return theta
