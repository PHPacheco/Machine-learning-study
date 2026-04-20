import matplotlib.pyplot as plt
import numpy as np

from Regressão.fase2.fase2_utils import calcular_correlacoes, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]
    y_data = base[target].to_numpy(dtype=float)

    correlacoes = calcular_correlacoes(base)
    num_features = len(features)

    # Definimos o layout dos subplots
    fig, axes = plt.subplots(1, num_features, figsize=(6 * num_features, 5), squeeze=False)
    fig.suptitle(f"Exercicio d - Correlacao e Regressao Linear Simples ({target})", fontsize=14)

    for i, col in enumerate(features):
        x_data = base[col].to_numpy(dtype=float)
        corr = correlacoes[col]
        
        # Ajuste linear simples para a reta
        ang, lin = np.polyfit(x_data, y_data, 1)
        
        ax = axes[0, i]
        ax.scatter(x_data, y_data, edgecolors="blue", facecolors="none")
        ax.plot(x_data, lin + ang * x_data, "r-")
        ax.set_title(f"{col} x {target}\ncorrelacao = {corr:.4f}")
        ax.set_xlabel(col)
        ax.set_ylabel(target)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    print(f"d) Graficos de dispersao para {num_features} variaveis gerados.")
    return fig, correlacoes


if __name__ == "__main__":
    executar()
    plt.show()
