import matplotlib.pyplot as plt

from fase2_utils import calcular_theta, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    tamanho = base["tamanho"].to_numpy(dtype=float)
    quartos = base["quartos"].to_numpy(dtype=float)
    preco = base["preco"].to_numpy(dtype=float)
    theta = calcular_theta(base).flatten()
    t0, t1, t2 = theta
    y_reg = t0 + t1 * tamanho + t2 * quartos

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Exercicio f - Linha de Regressao no Grafico 3D", fontsize=14)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tamanho, quartos, preco, edgecolors="blue", facecolors="none", s=50)
    ax.plot(tamanho, quartos, y_reg, "r-", linewidth=2)
    ax.set_xlabel("Tamanho da casa")
    ax.set_ylabel("Numero de quartos")
    ax.set_zlabel("Preco da casa")
    ax.view_init(elev=22, azim=45)

    print("f) Linha de regressao tracada no grafico 3D.")
    return fig, ax


if __name__ == "__main__":
    executar()
    plt.show()
