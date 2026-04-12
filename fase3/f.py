import matplotlib.pyplot as plt
import numpy as np

from c import calcular_polinomio
from e import QuestaoE


class QuestaoF:
    def executar(self, exibir=True):
        x, y = QuestaoE().executar()
        beta8 = np.polyfit(x, y, 8)
        y8 = calcular_polinomio(beta8, x)
        plt.plot(x, y8, "y", label="Regressão N=8")

        plt.xlabel("x")
        plt.ylabel("y")
        plt.title("Regressão Polinomial - Fase 3")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)

        if exibir:
            print("Exibindo o gráfico de regressão polinomial...")
            plt.show()
        return x, y


if __name__ == "__main__":
    QuestaoF().executar(exibir=True)
