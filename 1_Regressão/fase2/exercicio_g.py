import matplotlib.pyplot as plt
import numpy as np

from fase2_utils import calcular_correlacoes, calcular_theta, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]
    num_features = len(features)
    
    correlacoes = calcular_correlacoes(base)
    theta = calcular_theta(base).flatten()

    fig, ax = None, None
    if num_features == 2:
        x1_data = base[features[0]].to_numpy(dtype=float)
        x2_data = base[features[1]].to_numpy(dtype=float)
        y_data = base[target].to_numpy(dtype=float)

        grade_x1 = np.linspace(x1_data.min(), x1_data.max(), 20)
        grade_x2 = np.linspace(x2_data.min(), x2_data.max(), 20)
        malha_x1, malha_x2 = np.meshgrid(grade_x1, grade_x2)
        malha_y = theta[0] + theta[1] * malha_x1 + theta[2] * malha_x2

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle(f"Exercicio g - Correlacoes e Superficie 3D ({target})", fontsize=14)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x1_data, x2_data, y_data, edgecolors="blue", facecolors="none", s=50)
        ax.plot_surface(malha_x1, malha_x2, malha_y, alpha=0.35, cmap="Reds")
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])    
        ax.set_zlabel(target)
        
        titulo_corrs = "Regressao Multipla\n"
        for col, corr in correlacoes.items():
            titulo_corrs += f"corr({col}, {target}) = {corr:.4f}\n"
        ax.set_title(titulo_corrs)
    else:
        print(f"g) Pulando o grafico 3D (exige exatamente 2 variaveis, dataset possui {num_features}).")
        print("Correlacoes calculadas:")
        for col, corr in correlacoes.items():
            print(f"  {col} vs {target}: {corr:.4f}")

    print("g) Coeficientes de correlacao processados.")
    return fig, ax, correlacoes


if __name__ == "__main__":
    executar()
    plt.show()
