import matplotlib.pyplot as plt
import numpy as np

from fase2_utils import calcular_theta, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    tamanho = base["tamanho"].to_numpy(dtype=float)
    quartos = base["quartos"].to_numpy(dtype=float)
    preco = base["preco"].to_numpy(dtype=float)
    theta = calcular_theta(base).flatten()

    grade_tamanho = np.linspace(tamanho.min(), tamanho.max(), 20)
    grade_quartos = np.linspace(quartos.min(), quartos.max(), 20)
    malha_tamanho, malha_quartos = np.meshgrid(grade_tamanho, grade_quartos)
    malha_preco = theta[0] + theta[1] * malha_tamanho + theta[2] * malha_quartos

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Exercicio f - Plano de Regressao no Grafico 3D", fontsize=14)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tamanho, quartos, preco, edgecolors="blue", facecolors="none", s=50)
    ax.plot_surface(malha_tamanho, malha_quartos, malha_preco, alpha=0.35, cmap="Reds")
    ax.set_xlabel("Tamanho")
    ax.set_ylabel("Quartos")
    ax.set_zlabel("Preco")
    ax.view_init(elev=22, azim=45)

    print("f) Plano de regressao tracado no grafico 3D.")
    return fig, ax


if __name__ == "__main__":
    executar()
    plt.show()
