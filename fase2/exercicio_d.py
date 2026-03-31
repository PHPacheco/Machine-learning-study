import matplotlib.pyplot as plt
import numpy as np

from fase2_utils import calcular_correlacoes, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    tamanho = base["tamanho"].to_numpy(dtype=float)
    quartos = base["quartos"].to_numpy(dtype=float)
    preco = base["preco"].to_numpy(dtype=float)

    corr_tamanho, corr_quartos = calcular_correlacoes(base)
    ang_tamanho, lin_tamanho = np.polyfit(tamanho, preco, 1)
    ang_quartos, lin_quartos = np.polyfit(quartos, preco, 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Exercicio d - Correlacao e Regressao Linear Simples", fontsize=14)

    axes[0].scatter(tamanho, preco, edgecolors="blue", facecolors="none")
    axes[0].plot(tamanho, lin_tamanho + ang_tamanho * tamanho, "r-")
    axes[0].set_title(f"Tamanho x Preco\ncorrelacao = {corr_tamanho:.4f}")
    axes[0].set_xlabel("Tamanho")
    axes[0].set_ylabel("Preco")

    axes[1].scatter(quartos, preco, edgecolors="blue", facecolors="none")
    axes[1].plot(quartos, lin_quartos + ang_quartos * quartos, "r-")
    axes[1].set_title(f"Quartos x Preco\ncorrelacao = {corr_quartos:.4f}")
    axes[1].set_xlabel("Quartos")
    axes[1].set_ylabel("Preco")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    print("d) Graficos de dispersao com correlacao e regressao gerados.")
    return fig, (corr_tamanho, corr_quartos)


if __name__ == "__main__":
    executar()
    plt.show()
