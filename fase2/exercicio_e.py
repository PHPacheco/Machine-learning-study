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
    theta = calcular_theta(base)
    beta0, beta1, beta2 = theta.flatten()

    fig = plt.figure(figsize=(10, 7))
    fig.suptitle("Exercicio e - Dispersao 3D da Regressao Multipla", fontsize=14)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(tamanho, quartos, preco, edgecolors="blue", facecolors="none", s=50)
    ax.set_xlabel("Tamanho")
    ax.set_ylabel("Quartos")
    ax.set_zlabel("Preco")

    print("e) Regressao multipla calculada.")
    print("Coeficientes da regressao multipla:")
    print(f"beta0 (intercepto): {beta0:.2f}")
    print(f"beta1 (tamanho):    {beta1:.6f}")
    print(f"beta2 (quartos):    {beta2:.6f}")
    print(
        "Equacao do modelo: "
        f"preco = {beta0:.2f} + ({beta1:.6f} * tamanho) + ({beta2:.6f} * quartos)"
    )
    return fig, ax, theta


if __name__ == "__main__":
    executar()
    plt.show()
