import matplotlib.pyplot as plt
import numpy as np

from fase2_utils import calcular_theta, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]
    num_features = len(features)
    
    # Prepara dados para regressao
    x_base = base[features].to_numpy(dtype=float)
    y_data = base[target].to_numpy(dtype=float)
    
    theta = calcular_theta(base).flatten()
    t0 = theta[0]
    outros_thetas = theta[1:]
    
    # y_reg = t0 + sum(ti * xi)
    y_reg = t0 + np.dot(x_base, outros_thetas)

    fig, ax = None, None
    if num_features == 2:
        x1_data = x_base[:, 0]
        x2_data = x_base[:, 1]

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle(f"Exercicio f - Linha de Regressao no Grafico 3D ({target})", fontsize=14)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x1_data, x2_data, y_data, edgecolors="blue", facecolors="none", s=50)
        ax.plot(x1_data, x2_data, y_reg, "r-", linewidth=2)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(target)
        ax.view_init(elev=22, azim=45)
    else:
        print(f"f) Pulando o grafico 3D (exige exatamente 2 variaveis, dataset possui {num_features}).")

    print(f"f) Linha de regressao para {num_features} variaveis calculada.")
    return fig, ax


if __name__ == "__main__":
    executar()
    plt.show()
