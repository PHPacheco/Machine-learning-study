from a import QuestaoA
from fase3_utils import GRAUS, calcular_metricas


class QuestaoK:
    def executar(self):
        x, y = QuestaoA().executar()
        from fase3_utils import dividir_treino_teste

        x_treino, x_teste, y_treino, y_teste = dividir_treino_teste(x, y)
        resultados = []

        for grau in GRAUS:
            resultados.append(
                calcular_metricas(grau, x_treino, y_treino, x_teste, y_teste)
            )

        print("k) Calculo do R2 para treino e teste:")
        for resultado in resultados:
            print(
                f"N={resultado['grau']}: "
                f"MSE treino={resultado['mse_treino']:.6f}, "
                f"MSE teste={resultado['mse_teste']:.6f}, "
                f"R2 treino={resultado['r2_treino']:.6f}, "
                f"R2 teste={resultado['r2_teste']:.6f}"
            )

        melhor_r2 = max(resultados, key=lambda item: item["r2_teste"])
        print(
            "Conclusao: quanto mais proximo de 1 estiver o R2, melhor o modelo explica "
            "a variacao dos dados. Se o R2 de treino for muito alto e o de teste cair, "
            "isso indica overfitting."
        )
        print(
            f"Neste experimento, o maior R2 de teste foi no modelo N={melhor_r2['grau']} "
            f"com R2 teste={melhor_r2['r2_teste']:.6f}."
        )
        return resultados


if __name__ == "__main__":
    QuestaoK().executar()
