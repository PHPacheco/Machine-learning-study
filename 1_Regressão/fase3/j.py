import numpy as np
from fase3_utils import (
    GRAUS,
    adicionar_curva,
    ajustar_modelo,
    calcular_mse,
    calcular_polinomio,
    calcular_r2,
    carregar_dados,
    criar_grafico_base,
    dividir_treino_teste,
    finalizar_grafico,
)

class QuestaoJ:
    def _calcular_metricas_teste(self, grau, x_treino, y_treino, x_teste, y_teste):
        coeficientes  = np.polyfit(x_treino, y_treino, grau)
        y_prev_teste  = calcular_polinomio(coeficientes, x_teste)
        return {
            "grau":         grau,
            "coeficientes": coeficientes,
            "mse_teste":    calcular_mse(y_teste, y_prev_teste),
            "r2_teste":     calcular_r2(y_teste, y_prev_teste),
        }

    def _imprimir_tabela(self, resultados):
        melhor = min(resultados, key=lambda r: r["mse_teste"])

        print("\n" + "=" * 50)
        print(f"{'Grau':>6}  {'MSE Teste':>14}  {'R² Teste':>10}")
        print("-" * 50)
        for r in resultados:
            marca = " <- Melhor" if r["grau"] == melhor["grau"] else ""
            print(
                f"{r['grau']:>6}  "
                f"{r['mse_teste']:>14.4f}  "
                f"{r['r2_teste']:>10.4f}"
                f"{marca}"
            )
        print("=" * 50)
        print(f"\nMais preciso (menor MSE no teste): Grau {melhor['grau']}")
        print(f"MSE = {melhor['mse_teste']:.4f}  |  R² = {melhor['r2_teste']:.4f}")
        return melhor

    def executar(self, exibir=True):
        x, y = carregar_dados()
        x_treino, x_teste, y_treino, y_teste = dividir_treino_teste(x, y)

        x_base = np.linspace(x.min(), x.max(), 300)

        resultados = []
        for grau in GRAUS:
            metricas        = self._calcular_metricas_teste(grau, x_treino, y_treino, x_teste, y_teste)
            modelo          = ajustar_modelo(grau, x_treino, y_treino, x_base)
            metricas["y_base"] = modelo["y_base"]
            resultados.append(metricas)

        self._imprimir_tabela(resultados)

        if exibir:
            fig, ax = criar_grafico_base(x, y)
            for r in resultados:
                adicionar_curva(ax, x_base, r["y_base"], r["grau"])
            finalizar_grafico(ax)


if __name__ == "__main__":
    QuestaoJ().executar(exibir=True)