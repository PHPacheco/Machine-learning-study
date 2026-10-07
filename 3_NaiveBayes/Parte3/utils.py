from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ml_studies.loaders import load_credit_risk_csv


class RiscoCredito:
    """Compatibilidade com o notebook, usando o CSV bruto como fonte."""

    COLUNAS: ClassVar[list[str]] = ["historia", "divida", "garantias", "renda"]

    @staticmethod
    def carregar(caminho: str | Path) -> pd.DataFrame:
        X, y = load_credit_risk_csv(caminho)
        frame = pd.DataFrame(X, columns=RiscoCredito.COLUNAS)
        frame["risco"] = y
        return frame

    @staticmethod
    def remover_moderado(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["risco"] != "moderado"].reset_index(drop=True)

    @staticmethod
    def encode_previsores(frame: pd.DataFrame) -> np.ndarray:
        return frame[RiscoCredito.COLUNAS].to_numpy(dtype=float)

    @staticmethod
    def encode_classe(frame: pd.DataFrame):
        encoder = LabelEncoder()
        return encoder.fit_transform(frame["risco"]), encoder

    @staticmethod
    def novo_cliente(historia: int, divida: int, garantias: int, renda: int) -> np.ndarray:
        return np.asarray([[historia, divida, garantias, renda]])
