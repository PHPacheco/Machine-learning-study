import numpy as np

from Regressão.fase2.fase2_utils import calcular_theta, carregar_base, montar_X_y
from sklearn.linear_model import LinearRegression


def executar(base=None):
    if base is None:
        base = carregar_base()

    features = base.columns[:-1].tolist()
    target = base.columns[-1]

    x_base, _, y = montar_X_y(base)
    theta_manual = calcular_theta(base).flatten()

    modelo = LinearRegression()
    modelo.fit(x_base, y)
    
    # Montamos o theta do sklearn: [intercepto, coef1, coef2, ...]
    intercepto = modelo.intercept_
    if isinstance(intercepto, np.ndarray):
        intercepto = intercepto.item()
        
    theta_sklearn = np.hstack(([intercepto], modelo.coef_.flatten()))

    print(f"i) Comparacao com scikit-learn ({target}):")
    print("Metodo: Equacao Normal (Manual)")
    print(f"beta0 (intercepto): {theta_manual[0]:.4f}")
    for i, col in enumerate(features):
        print(f"beta{i+1} ({col}): {theta_manual[i+1]:.6f}")
    
    print("\nMetodo: Scikit-learn (Otimizado)")
    print(f"beta0 (intercepto): {theta_sklearn[0]:.4f}")
    for i, col in enumerate(features):
        print(f"beta{i+1} ({col}): {theta_sklearn[i+1]:.6f}")
        
    return theta_manual, theta_sklearn


if __name__ == "__main__":
    executar()
