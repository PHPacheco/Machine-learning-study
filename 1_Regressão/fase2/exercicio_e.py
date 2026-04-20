import matplotlib.pyplot as plt
import numpy as np

from Regressão.fase2.fase2_utils import calcular_theta, carregar_base


plt.style.use("seaborn-v0_8-whitegrid")


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]
    theta = calcular_theta(base)
    theta_flat = theta.flatten()
    
    beta0 = theta_flat[0]
    outros_betas = theta_flat[1:]

    num_features = len(features)
    
    fig, ax = None, None
    if num_features == 2:
        x1_data = base[features[0]].to_numpy(dtype=float)
        x2_data = base[features[1]].to_numpy(dtype=float)
        y_data = base[target].to_numpy(dtype=float)

        fig = plt.figure(figsize=(10, 7))
        fig.suptitle(f"Exercicio e - Dispersao 3D da Regressao Multipla ({target})", fontsize=14)
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x1_data, x2_data, y_data, edgecolors="blue", facecolors="none", s=50)
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel(target)
    else:
        print(f"e) Pulando o grafico 3D (exige exatamente 2 variaveis, dataset possui {num_features}).")

    print("e) Regressao multipla calculada.")
    print("Coeficientes da regressao multipla:")
    print(f"beta0 (intercepto): {beta0:.4f}")
    
    equacao = f"{target} = {beta0:.4f}"
    for i, col in enumerate(features):
        beta_i = outros_betas[i]
        print(f"beta{i+1} ({col}): {beta_i:.6f}")
        equacao += f" + ({beta_i:.6f} * {col})"
        
    print(f"Equacao do modelo: {equacao}")
    
    return fig, ax, theta


if __name__ == "__main__":
    executar()
    plt.show()
