import numpy as np

from fase2_utils import calcular_theta, carregar_base


def executar(base=None):
    if base is None:
        base = carregar_base()

    theta = calcular_theta(base).flatten()
    beta0, beta1, beta2 = theta
    preco_3_quartos = float(np.array([1, 1650, 3]) @ theta)
    preco_2_quartos = float(np.array([1, 1650, 2]) @ theta)
    preco_4_quartos = float(np.array([1, 1650, 4]) @ theta)

    print("h) Previsao para casa com 1650 de tamanho:")
    print("Coeficientes usados na previsao:")
    print(f"beta0 (intercepto): {beta0:.2f}")
    print(f"beta1 (tamanho):    {beta1:.6f}")
    print(f"beta2 (quartos):    {beta2:.6f}")
    print(
        "Equacao do modelo: "
        f"preco = {beta0:.2f} + ({beta1:.6f} * tamanho) + ({beta2:.6f} * quartos)"
    )
    print()
    print(f"3 quartos: {preco_3_quartos:.2f}")
    print(f"2 quartos: {preco_2_quartos:.2f}")
    print(f"4 quartos: {preco_4_quartos:.2f}")
   
    print(
        "Ao aumentar a quantidade de quartos, o preco previsto cai; "
        "ao diminuir, ele sobe. Isso acontece porque, mantendo o "
        "tamanho fixo, o coeficiente de quartos na regressao ficou "
        "negativo neste conjunto de dados."
    )
    return preco_3_quartos, preco_2_quartos, preco_4_quartos


if __name__ == "__main__":
    executar()
