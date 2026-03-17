import math

import matplotlib.pyplot as plt

class AnaliseLinear:

    def correlacao(self, x, y):

        n = len(x)
        soma_x = sum(x)
        soma_y = sum(y)
        soma_xy = sum(xi * yi for xi, yi in zip(x, y))
        soma_x2 = sum(xi * xi for xi in x)
        soma_y2 = sum(yi * yi for yi in y)

        numerador = n * soma_xy - soma_x * soma_y
        denominador = math.sqrt((n * soma_x2 - soma_x**2) * (n * soma_y2 - soma_y**2))

        if denominador == 0:
            raise ValueError("Nao e possivel calcular correlacao: denominador igual a zero")

        return numerador / denominador

    def regressao(self, x, y):

        n = len(x)
        soma_x = sum(x)
        soma_y = sum(y)
        soma_xy = sum(xi * yi for xi, yi in zip(x, y))
        soma_x2 = sum(xi * xi for xi in x)

        denominador = n * soma_x2 - soma_x**2
        if denominador == 0:
            raise ValueError("Nao e possivel calcular regressao: denominador igual a zero")

        beta1 = (n * soma_xy - soma_x * soma_y) / denominador
        beta0 = (soma_y - beta1 * soma_x) / n

        return beta0, beta1


def plotar_dispersao(x, y, titulo):
    plt.figure(figsize=(6, 4))
    plt.scatter(x, y)
    plt.title(titulo)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)


def main():
    x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

    x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

    x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
    y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]

    x4 = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
    y4 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]

    dados = {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "x3": x3,
        "y3": y3,
        "x4": x4,
        "y4": y4,
    }

    analise = AnaliseLinear()

    grupos = [("x1", "y1"), ("x2", "y2"), ("x3", "y3"), ("x4", "y4")]

    for gx, gy in grupos:
        x = dados[gx]
        y = dados[gy]
        r = analise.correlacao(x, y)
        beta0, beta1 = analise.regressao(x, y)

        print(f"{gx}-{gy}: r={r:.6f}, beta0={beta0:.6f}, beta1={beta1:.6f}")
        plotar_dispersao(x, y, f"Dispersao {gx}-{gy}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
