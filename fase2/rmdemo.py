import numpy as np

from regmultipla import regmultipla


def carregar_dados():
    try:
        from scipy.io import loadmat  # type: ignore

        mat = loadmat("data.mat")
        for valor in mat.values():
            if isinstance(valor, np.ndarray) and valor.ndim == 2 and valor.shape[1] >= 3:
                return valor[:, :3].astype(float)
    except Exception:
        pass

    return np.loadtxt("C:/Users/pedro/OneDrive/Documentos/AprendizadoDeMaquina/trabalho1/fase2/data.csv", delimiter=",")


def main():
    dados = carregar_dados()

    X_base = dados[:, :2]
    y = dados[:, 2]
    m = X_base.shape[0]
    X = np.column_stack([np.ones(m), X_base])

    theta = regmultipla(X, y)

    print("Parametros estimados (theta):")
    print(f"theta0 = {theta[0, 0]:.6f}")
    print(f"theta1 = {theta[1, 0]:.6f}")
    print(f"theta2 = {theta[2, 0]:.6f}")

    # Exemplo de previsao para casa com 1650 pes e 3 quartos.
    x_novo = np.array([1, 1650, 3], dtype=float)
    preco_previsto = float(x_novo @ theta[:, 0])
    print(f"\nPreco previsto para casa de 1650 pes e 3 quartos: {preco_previsto:.2f}")


if __name__ == "__main__":
    main()
