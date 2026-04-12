import matplotlib.pyplot as plt

from a import QuestaoA


class QuestaoB:
    def executar(self, exibir=False):
        x, y = QuestaoA().executar()
        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color="blue", label="Dados Originais", alpha=0.6)
        if exibir:
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Regressão Polinomial - Fase 3")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.show()
        return x, y


if __name__ == "__main__":
    QuestaoB().executar(exibir=True)
