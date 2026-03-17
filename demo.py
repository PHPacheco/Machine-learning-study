import matplotlib.pyplot as plt

from ex01 import AnaliseLinear


def main():
    x1 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y1 = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

    x2 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    y2 = [9.14, 8.14, 8.47, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]

    x3 = [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 19]
    y3 = [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 5.56, 7.91, 6.89, 12.50]

    x4 = [10.0, 8.0, 13.0, 9.0, 11.0, 14.0, 6.0, 4.0, 12.0, 7.0, 5.0]
    y4 = [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]

    datasets = [
        ("Dataset 1", x1, y1),
        ("Dataset 2", x2, y2),
        ("Dataset 3", x3, y3),
        ("Dataset 4", x4, y4),
    ]

    analise = AnaliseLinear()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, (nome, x, y) in enumerate(datasets):
        r = analise.correlacao(x, y)
        beta0, beta1 = analise.regressao(x, y)

        x_linha = [min(x), max(x)]
        y_linha = [beta0 + beta1 * xv for xv in x_linha]

        axes[i].scatter(x, y, color="tab:blue")
        axes[i].plot(x_linha, y_linha, color="tab:red", linewidth=2)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(
            f"{nome} | r={r:.4f} | beta0={beta0:.4f} | beta1={beta1:.4f}"
        )

        print(f"{nome}: r={r:.6f}, beta0={beta0:.6f}, beta1={beta1:.6f}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
