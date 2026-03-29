import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

from regmultipla import regmultipla

# estilo visual
plt.style.use("seaborn-v0_8-whitegrid")


def carregar_dados():
    try:
        from scipy.io import loadmat

        for caminho in ["fase2/data.mat", "data.mat"]:
            if Path(caminho).exists():
                mat = loadmat(caminho)
                for valor in mat.values():
                    if isinstance(valor, np.ndarray) and valor.ndim == 2 and valor.shape[1] >= 3:
                        return valor[:, :3].astype(float)
    except Exception:
        pass

    for caminho in ["fase2/data.csv", "data.csv"]:
        if Path(caminho).exists():
            return np.loadtxt(caminho, delimiter=",")

    raise FileNotFoundError("Nao encontrei data.csv nem data.mat")


def correlacao(x, y):
    return np.corrcoef(x, y)[0, 1]


def regressao_simples(x, y):
    b1, b0 = np.polyfit(x, y, 1)
    return b0, b1


def main():
    dados = carregar_dados()

    X_base = dados[:, :2]
    y = dados[:, 2]

    tamanho = X_base[:, 0]
    quartos = X_base[:, 1]

    m = X_base.shape[0]
    X = np.column_stack([np.ones(m), X_base])

    theta = regmultipla(X, y)

    r_tam = correlacao(tamanho, y)
    r_qua = correlacao(quartos, y)

    b0_tam, b1_tam = regressao_simples(tamanho, y)
    b0_qua, b1_qua = regressao_simples(quartos, y)

    print("Parametros estimados (theta):")
    print(f"theta0 = {theta[0, 0]:.6f}")
    print(f"theta1 = {theta[1, 0]:.6f}")
    print(f"theta2 = {theta[2, 0]:.6f}")

    print("\nCorrelacoes:")
    print(f"Tamanho x Preco: {r_tam:.6f}")
    print(f"Quartos x Preco: {r_qua:.6f}")

    print("\nEquacao da regressao multipla:")
    print(
        f"Preco = {theta[0, 0]:.6f} + "
        f"{theta[1, 0]:.6f} * Tamanho + "
        f"{theta[2, 0]:.6f} * Quartos"
    )

    # ========================
    # GRAFICO 1
    # ========================
    plt.figure(figsize=(6, 4))

    plt.scatter(
        tamanho,
        y,
        facecolors="none",
        edgecolors="blue",
        s=40
    )

    x_linha = np.linspace(tamanho.min(), tamanho.max(), 100)
    plt.plot(x_linha, b0_tam + b1_tam * x_linha, "r-", linewidth=1.5)

    plt.title(f"Coeficiente: {r_tam:.4f}", fontsize=10, fontweight="bold")
    plt.xlabel("Tamanho da Casa", fontsize=9)
    plt.ylabel("Preco por R$ 10.000", fontsize=9)

    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.grid(alpha=0.3)

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

    # Pontos de dispersao
    ax.scatter(
        tamanho,
        quartos,
        y,
        facecolors="none",
        edgecolors="blue",
        s=40,
        zorder=5
    )

    # regressao multipla
    tam_grid = np.linspace(tamanho.min(), tamanho.max(), 30)
    qua_grid = np.linspace(quartos.min(), quartos.max(), 30)
    TAM, QUA = np.meshgrid(tam_grid, qua_grid)
    Z_PLANO = theta[0, 0] + theta[1, 0] * TAM + theta[2, 0] * QUA

    ax.plot_surface(
        TAM, QUA, Z_PLANO,
        alpha=0.35,
        color="red",
        edgecolor="none"
    )

    ax.plot_wireframe(
        TAM, QUA, Z_PLANO,
        alpha=0.15,
        color="darkred",
        linewidth=0.5
    )
    # Fim plano de regressão

    ax.set_xlabel("Tamanho da Casa", fontsize=9, labelpad=10)
    ax.set_ylabel("Numero de Quartos", fontsize=9, labelpad=10)
    ax.set_zlabel("Preco por R$ 10.000", fontsize=9, labelpad=10)

    titulo = (
        f"Regressao Multipla  |  "
        f"r(Tam, Preco) = {r_tam:.4f}   "
        f"r(Qua, Preco) = {r_qua:.4f}\n"
        f"Coef: θ0={theta[0,0]:.2f}  θ1={theta[1,0]:.2f}  θ2={theta[2,0]:.2f}"
    )
    ax.set_title(titulo, fontsize=9, fontweight="bold", pad=15)

    ax.tick_params(axis="both", which="major", labelsize=8)

    ax.view_init(elev=25, azim=-60)

    x_novo = np.array([1, 1650, 3], dtype=float)
    preco_previsto = float(x_novo @ theta[:, 0])
    print(f"\nPreco previsto para casa de 1650 pes e 3 quartos: {preco_previsto:.2f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()