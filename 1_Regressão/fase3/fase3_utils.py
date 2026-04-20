import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as scipy

try:
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
except ImportError:
    mean_squared_error = None
    r2_score = None
    train_test_split = None


GRAUS = [1, 2, 3, 8]
CORES = {1: "r", 2: "g", 3: "k", 8: "y"}


def carregar_dados():
    base_dir = os.path.dirname(__file__)
    file_mat = os.path.join(base_dir, "data_preg.mat")
    file_csv = os.path.join(base_dir, "data_preg.csv")

    if os.path.exists(file_mat):
        mat = scipy.loadmat(file_mat)
        if "data" in mat:
            data = mat["data"]
        else:
            keys = [k for k in mat.keys() if not k.startswith("__")]
            data = mat[keys[0]]
        return data[:, 0], data[:, 1]

    data = pd.read_csv(file_csv, header=None)
    return data[0].values, data[1].values


def calcular_polinomio(coeffs, x_vals):
    n = len(coeffs) - 1
    y_vals = np.zeros_like(x_vals, dtype=float)
    for i, beta in enumerate(coeffs):
        y_vals += beta * (x_vals ** (n - i))
    return y_vals


def dividir_treino_teste(x, y, test_size=0.3, random_state=42):
    if train_test_split is not None:
        return train_test_split(x, y, test_size=test_size, random_state=random_state)

    rng = np.random.default_rng(random_state)
    indices = np.arange(len(x))
    rng.shuffle(indices)

    qtd_teste = int(np.ceil(len(indices) * test_size))
    indices_teste = indices[:qtd_teste]
    indices_treino = indices[qtd_teste:]

    return (
        x[indices_treino],
        x[indices_teste],
        y[indices_treino],
        y[indices_teste],
    )


def calcular_mse(y_real, y_prev):
    if mean_squared_error is not None:
        return mean_squared_error(y_real, y_prev)
    return float(np.mean((y_real - y_prev) ** 2))


def calcular_r2(y_real, y_prev):
    if r2_score is not None:
        return r2_score(y_real, y_prev)

    soma_residuos = np.sum((y_real - y_prev) ** 2)
    soma_total = np.sum((y_real - np.mean(y_real)) ** 2)
    return float(1 - (soma_residuos / soma_total))


def ajustar_modelo(grau, x_treino, y_treino, x_base=None):
    coeficientes = np.polyfit(x_treino, y_treino, grau)
    resultado = {"grau": grau, "coeficientes": coeficientes}
    if x_base is not None:
        resultado["y_base"] = calcular_polinomio(coeficientes, x_base)
    return resultado


def calcular_metricas(grau, x_treino, y_treino, x_teste, y_teste):
    coeficientes = np.polyfit(x_treino, y_treino, grau)
    y_prev_treino = calcular_polinomio(coeficientes, x_treino)
    y_prev_teste = calcular_polinomio(coeficientes, x_teste)
    return {
        "grau": grau,
        "coeficientes": coeficientes,
        "mse_treino": calcular_mse(y_treino, y_prev_treino),
        "mse_teste": calcular_mse(y_teste, y_prev_teste),
        "r2_treino": calcular_r2(y_treino, y_prev_treino),
        "r2_teste": calcular_r2(y_teste, y_prev_teste),
    }


def criar_grafico_base(x, y):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, color="blue", label="Dados Originais", alpha=0.6)
    return fig, ax


def adicionar_curva(ax, x, y_ajustado, grau):
    ordem = np.argsort(x)
    ax.plot(x[ordem], y_ajustado[ordem], CORES[grau], label=f"Regressao N={grau}")


def finalizar_grafico(ax):
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Regressao Polinomial - Fase 3")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.show()
