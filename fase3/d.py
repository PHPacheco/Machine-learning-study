import matplotlib.pyplot as plt
import numpy as np

from c import QuestaoC, calcular_polinomio


class QuestaoD:
    def executar(self, exibir=False):
        x, y = QuestaoC().executar()
        beta2 = np.polyfit(x, y, 2)
        y2 = calcular_polinomio(beta2, x)
        plt.plot(x, y2, "g", label="Regressão N=2")
        if exibir:
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Regressão Polinomial - Fase 3")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()
        return x, y


if __name__ == "__main__":
    QuestaoD().executar(exibir=True)
