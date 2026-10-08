"""Exemplo mínimo das funções reutilizáveis de regressão."""

from pathlib import Path

from ml_studies.regression import (
    design_matrix,
    feature_correlations,
    load_regression_frame,
    multiple_linear_regression,
)


def main() -> None:
    dados = Path(__file__).with_name("dados")
    frame = load_regression_frame(dados)
    _, matriz, alvo = design_matrix(frame)
    coeficientes = multiple_linear_regression(matriz, alvo).reshape(-1)

    print("correlações:", feature_correlations(frame))
    print("coeficientes [intercepto, tamanho, quartos]:", coeficientes.tolist())


if __name__ == "__main__":
    main()
