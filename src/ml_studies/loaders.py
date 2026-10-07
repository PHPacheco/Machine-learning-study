"""Loaders explícitos para fontes tabulares e artefatos legados."""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import OrdinalEncoder


def load_credit_risk_csv(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.str_]]:
    frame = pd.read_csv(Path(path))
    if frame.shape[1] < 2:
        raise ValueError("o CSV precisa de previsores e uma coluna-alvo")
    X_raw = frame.iloc[:, :-1].astype(str)
    y = frame.iloc[:, -1].astype(str).to_numpy()
    X = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit_transform(X_raw)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=np.str_)


def load_trusted_legacy_pickle(path: str | Path, *, trusted: bool = False) -> Any:
    """Carrega pickle somente após confirmação explícita da origem.

    Pickle pode executar código arbitrário durante a desserialização. Nunca use
    esta função com downloads ou arquivos cuja origem/integridade não foi
    verificada.
    """

    if not trusted:
        raise ValueError("pickle bloqueado: confirme a origem com trusted=True")
    warnings.warn(
        "carregando pickle legado confiável; prefira reconstruir a partir de dados brutos",
        UserWarning,
        stacklevel=2,
    )
    with Path(path).open("rb") as stream:
        return pickle.load(stream)
