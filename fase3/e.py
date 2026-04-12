import matplotlib.pyplot as plt
import numpy as np

from c import calcular_polinomio
from d import QuestaoD


class QuestaoE:
    def executar(self, exibir=False):
        x, y = QuestaoD().executar()
        beta3 = np.polyfit(x, y, 3)
        y3 = calcular_polinomio(beta3, x)
        plt.plot(x, y3, "k", label="Regressão N=3")
        if exibir:
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Regressão Polinomial - Fase 3")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()
        return x, y


if __name__ == "__main__":
    QuestaoE().executar(exibir=True)
