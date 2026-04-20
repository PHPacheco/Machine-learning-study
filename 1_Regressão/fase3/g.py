"""
Regressão Polinomial com EQM — usa utilitários de utils_regressao.py
=====================================================================
Ajusta modelos polinomiais de graus definidos em GRAUS, calcula o
Erro Quadrático Médio (treino e teste) e exibe os resultados.
"""

import numpy as np
from fase3_utils import (
    GRAUS,
    adicionar_curva,
    ajustar_modelo,
    calcular_metricas,
    carregar_dados,
    criar_grafico_base,
    dividir_treino_teste,
    finalizar_grafico,
)

class QuestaoG:
    def executar(self, exibir=True):

        x, y = carregar_dados()
        x_treino, x_teste, y_treino, y_teste = dividir_treino_teste(x, y)

        x_base = np.linspace(x.min(), x.max(), 300)

        resultados = []
        for grau in GRAUS:
            metricas = calcular_metricas(grau, x_treino, y_treino, x_teste, y_teste)
            modelo   = ajustar_modelo(grau, x_treino, y_treino, x_base)
            metricas["y_base"] = modelo["y_base"]
            resultados.append(metricas)

        print("\n" + "=" * 72)
        print(f"{'Grau':>6}  {'MSE Treino':>14}  {'MSE Teste':>14}  {'R² Treino':>10}  {'R² Teste':>10}")
        print("-" * 72)

        melhor = min(resultados, key=lambda r: r["mse_teste"])

        for r in resultados:
            marca = " <- " if r["grau"] == melhor["grau"] else "  "
            print(
                f"{r['grau']:>6}  "
                f"{r['mse_treino']:>14.4f}  "
                f"{r['mse_teste']:>14.4f}  "
                f"{r['r2_treino']:>10.4f}  "
                f"{r['r2_teste']:>10.4f}"
                f"{marca}"
            )

        print("=" * 72)
        print(f"\nModelo mais preciso (menor MSE no teste): Grau {melhor['grau']}")
        print(f"MSE Teste = {melhor['mse_teste']:.4f}  |  R² Teste = {melhor['r2_teste']:.4f}")

        print(f"\nCoeficientes do grau {melhor['grau']}:")
        n = melhor["grau"]
        for i, beta in enumerate(melhor["coeficientes"]):
            exp = n - i
            nome = f"β{exp}" if exp > 0 else "β₀"
            print(f"{nome:>4} = {beta:+.6f}")



        fig, ax = criar_grafico_base(x, y)

        for r in resultados:
            adicionar_curva(ax, x_base, r["y_base"], r["grau"])

        finalizar_grafico(ax)

if __name__ == "__main__":
    QuestaoG().executar(exibir=True)
