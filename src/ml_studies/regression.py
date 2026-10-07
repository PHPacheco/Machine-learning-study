"""Funções reutilizáveis para os estudos de regressão."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


def _paired_vectors(x: ArrayLike, y: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x_array = np.asarray(x, dtype=float).reshape(-1)
    y_array = np.asarray(y, dtype=float).reshape(-1)
    if len(x_array) != len(y_array) or len(x_array) < 2:
        raise ValueError("x e y devem ter o mesmo tamanho e pelo menos duas amostras")
    return x_array, y_array


@dataclass(frozen=True, slots=True)
class LinearAnalysis:
    """Correlação de Pearson e regressão linear simples."""

    def correlation(self, x: ArrayLike, y: ArrayLike) -> float:
        x_array, y_array = _paired_vectors(x, y)
        if np.isclose(np.std(x_array), 0) or np.isclose(np.std(y_array), 0):
            raise ValueError("não é possível calcular correlação com variância zero")
        return float(np.corrcoef(x_array, y_array)[0, 1])

    def fit(self, x: ArrayLike, y: ArrayLike) -> tuple[float, float]:
        x_array, y_array = _paired_vectors(x, y)
        design = np.column_stack([np.ones(len(x_array)), x_array])
        coefficients, *_ = np.linalg.lstsq(design, y_array, rcond=None)
        return float(coefficients[0]), float(coefficients[1])

    # Compatibilidade com os nomes usados no material original.
    correlacao = correlation
    regressao = fit


AnaliseLinear = LinearAnalysis


def multiple_linear_regression(X: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    design = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float).reshape(-1, 1)
    if design.ndim != 2 or design.shape[0] != target.shape[0]:
        raise ValueError("X deve ser 2D e ter o mesmo número de linhas que y")
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return coefficients


def load_regression_frame(directory: str | Path) -> pd.DataFrame:
    root = Path(directory)
    csv_path = root / "data.csv"
    mat_path = root / "data.mat"
    if csv_path.is_file():
        values = pd.read_csv(csv_path, header=None)
    elif mat_path.is_file():
        from scipy.io import loadmat

        raw = loadmat(mat_path)
        matrices = [
            value
            for key, value in raw.items()
            if not key.startswith("__") and isinstance(value, np.ndarray) and value.ndim == 2
        ]
        if not matrices:
            raise ValueError(f"nenhuma matriz 2D encontrada em {mat_path}")
        values = pd.DataFrame(matrices[0])
    else:
        raise FileNotFoundError(f"data.csv ou data.mat não encontrado em {root}")

    count = values.shape[1]
    values.columns = (
        ["tamanho", "quartos", "preco"]
        if count == 3
        else [f"feat_{index + 1}" for index in range(count - 1)] + ["alvo"]
    )
    return values


def design_matrix(frame: pd.DataFrame) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any]]:
    X_base = frame.iloc[:, :-1].to_numpy(dtype=float)
    y = frame.iloc[:, -1].to_numpy(dtype=float).reshape(-1, 1)
    X = np.column_stack([np.ones(len(X_base)), X_base])
    return X_base, X, y


def feature_correlations(frame: pd.DataFrame) -> dict[str, float]:
    target = frame.columns[-1]
    return {
        column: float(np.corrcoef(frame[column], frame[target])[0, 1])
        for column in frame.columns[:-1]
    }
