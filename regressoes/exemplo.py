"""Exemplo curto de regressão linear múltipla."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    arquivo = Path(__file__).with_name("dados") / "data.csv"
    dados = pd.read_csv(
        arquivo,
        header=None,
        names=["tamanho", "quartos", "preco"],
    )

    atributos = dados[["tamanho", "quartos"]].to_numpy(dtype=float)
    alvo = dados["preco"].to_numpy(dtype=float)
    matriz = np.column_stack([np.ones(len(atributos)), atributos])
    coeficientes, *_ = np.linalg.lstsq(matriz, alvo, rcond=None)

    correlacoes = dados.corr(numeric_only=True)["preco"].drop("preco")
    print("Correlações com o preço:")
    print(correlacoes.round(4).to_string())
    print("\nCoeficientes [intercepto, tamanho, quartos]:")
    print(coeficientes.round(4).tolist())


if __name__ == "__main__":
    main()
