from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from Regressão.fase2.regmultipla import regmultipla


def carregar_base() -> pd.DataFrame:
    csv_path = Path(__file__).with_name("data.csv")
    mat_path = Path(__file__).with_name("data.mat")

    if csv_path.exists():
        df_temp = pd.read_csv(csv_path, header=None)
        num_cols = df_temp.shape[1]
        
        if num_cols == 3:
            colunas = ["tamanho", "quartos", "preco"]
        else:
            colunas = [f"feat_{i+1}" for i in range(num_cols - 1)] + ["preco"]
        
        df_temp.columns = colunas
        return df_temp

    if mat_path.exists():
        try:
            from scipy.io import loadmat  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "data.mat encontrado, mas scipy nao esta instalado para leitura do arquivo."
            ) from exc

        mat = loadmat(mat_path)
        for valor in mat.values():
            if isinstance(valor, np.ndarray) and valor.ndim == 2:
                num_cols = valor.shape[1]
                if num_cols == 3:
                    colunas = ["tamanho", "quartos", "preco"]
                else:
                    colunas = [f"feat_{i+1}" for i in range(num_cols - 1)] + ["preco"]
                return pd.DataFrame(valor, columns=colunas)

    raise FileNotFoundError("Nao encontrei data.csv nem data.mat na pasta fase2.")


def montar_X_y(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # x_base contem todas as colunas exceto a ultima
    x_base = base.iloc[:, :-1].to_numpy(dtype=float)
    # y contem apenas a ultima coluna
    y = base.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)
    
    # A coluna de 1s representa o intercepto (beta0) na equacao da regressao.
    intercepto = np.ones((x_base.shape[0], 1))
    x = np.hstack((intercepto, x_base))
    return x_base, x, y


def calcular_correlacoes(base: pd.DataFrame) -> dict[str, float]:
    # Calcula a correlacao de todas as colunas de entrada com a coluna de preco
    preco_col = base.columns[-1]
    correlacoes = {}
    for col in base.columns[:-1]:
        corr = float(np.corrcoef(base[col], base[preco_col])[0, 1])
        correlacoes[col] = corr
    return correlacoes


def calcular_theta(base: pd.DataFrame) -> np.ndarray:
    _, x, y = montar_X_y(base)
    return regmultipla(x, y)
