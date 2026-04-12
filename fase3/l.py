from k import QuestaoK


class QuestaoL:
    def executar(self, resultados=None):
        if resultados is None:
            resultados = QuestaoK().executar()
        melhor_modelo = min(resultados, key=lambda item: item["mse_teste"])
        print("l) Modelo mais preciso:")
        print(
            f"O modelo mais preciso neste caso e o de grau N={melhor_modelo['grau']}, "
            f"pois apresentou o menor erro no conjunto de teste "
            f"(MSE teste={melhor_modelo['mse_teste']:.6f}) e o melhor poder de "
            f"generalizacao (R2 teste={melhor_modelo['r2_teste']:.6f})."
        )
        return melhor_modelo


if __name__ == "__main__":
    QuestaoL().executar()
