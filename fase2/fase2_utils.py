from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from regmultipla import regmultipla


COLUNAS = ["tamanho", "quartos", "preco"]


def carregar_base() -> pd.DataFrame:
    csv_path = Path(__file__).with_name("data.csv")
    mat_path = Path(__file__).with_name("data.mat")

    if csv_path.exists():
        return pd.read_csv(csv_path, header=None, names=COLUNAS)

    if mat_path.exists():
        try:
            from scipy.io import loadmat  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "data.mat encontrado, mas scipy nao esta instalado para leitura do arquivo."
            ) from exc

        mat = loadmat(mat_path)
        for valor in mat.values():
            if isinstance(valor, np.ndarray) and valor.ndim == 2 and valor.shape[1] >= 3:
                return pd.DataFrame(valor[:, :3], columns=COLUNAS)

    raise FileNotFoundError("Nao encontrei data.csv nem data.mat na pasta fase2.")


def montar_X_y(base: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_base = base[["tamanho", "quartos"]].to_numpy(dtype=float)
    y = base["preco"].to_numpy(dtype=float).reshape(-1, 1)
    # A coluna de 1s representa o intercepto (beta0) na equacao da regressao.
    intercepto = np.ones((x_base.shape[0], 1))
    x = np.hstack((intercepto, x_base))
    return x_base, x, y


def calcular_correlacoes(base: pd.DataFrame) -> tuple[float, float]:
    preco = base["preco"].to_numpy(dtype=float)
    corr_tamanho = float(np.corrcoef(base["tamanho"], preco)[0, 1])
    corr_quartos = float(np.corrcoef(base["quartos"], preco)[0, 1])
    return corr_tamanho, corr_quartos


def calcular_theta(base: pd.DataFrame) -> np.ndarray:
    _, x, y = montar_X_y(base)
    return regmultipla(x, y)
