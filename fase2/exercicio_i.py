import numpy as np

from fase2_utils import calcular_theta, carregar_base, montar_X_y
from sklearn.linear_model import LinearRegression


def executar(base=None):
    if base is None:
        base = carregar_base()

    x_base, _, y = montar_X_y(base)
    theta_manual = calcular_theta(base).flatten()

    modelo = LinearRegression()
    modelo.fit(x_base, y)
    theta_sklearn = np.array(
        [modelo.intercept_.item(), modelo.coef_.flatten()[0], modelo.coef_.flatten()[1]]
    )

    print("i) Comparacao com scikit-learn:")
    print("Equacao normal:")
    print(f"beta0 (intercepto): {theta_manual[0]:.2f}")
    print(f"beta1 (tamanho):    {theta_manual[1]:.6f}")
    print(f"beta2 (quartos):    {theta_manual[2]:.6f}")
    print()
    print("Scikit-learn:")
    print(f"beta0 (intercepto): {theta_sklearn[0]:.2f}")
    print(f"beta1 (tamanho):    {theta_sklearn[1]:.6f}")
    print(f"beta2 (quartos):    {theta_sklearn[2]:.6f}")
    return theta_manual, theta_sklearn


if __name__ == "__main__":
    executar()
