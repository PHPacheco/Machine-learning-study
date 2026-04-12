import matplotlib.pyplot as plt
import numpy as np

from b import QuestaoB


def calcular_polinomio(coeffs, x_vals):
    n = len(coeffs) - 1
    y_vals = np.zeros_like(x_vals, dtype=float)
    for i, beta in enumerate(coeffs):
        y_vals += beta * (x_vals ** (n - i))
    return y_vals


class QuestaoC:
    def executar(self, exibir=False):
        x, y = QuestaoB().executar()
        beta1 = np.polyfit(x, y, 1)
        y1 = calcular_polinomio(beta1, x)
        plt.plot(x, y1, "r", label="Regressão N=1")
        if exibir:
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Regressão Polinomial - Fase 3")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()
        return x, y


if __name__ == "__main__":
    QuestaoC().executar(exibir=True)
