from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

class RiscoCredito:
    """Utilitários para carregar, pré-processar e modelar a base de risco de crédito."""

    # Mapeamentos dos valores codificados para facilitar leitura
    MAP_HISTORIA   = {0: 'ruim', 1: 'desconhecida', 2: 'boa'}
    MAP_DIVIDA     = {0: 'baixa', 1: 'alta'}
    MAP_GARANTIAS  = {0: 'nenhuma', 1: 'adequada'}
    MAP_RENDA      = {0: '< 15', 1: '15 a 35', 2: '> 35'}
    COLUNAS        = ['historia', 'divida', 'garantias', 'renda']

    @staticmethod
    def carregar(caminho: str) -> pd.DataFrame:
        """Carrega o arquivo .pkl e retorna um DataFrame completo."""
        import pickle
        from pathlib import Path

        caminho_arquivo = Path(caminho)

        if not caminho_arquivo.exists():
            caminho_arquivo = Path(__file__).parent / caminho_arquivo.name

        with open(caminho_arquivo, 'rb') as f:
            dados = pickle.load(f)

        X_raw, y_raw = dados[0], dados[1]
        df = pd.DataFrame(X_raw, columns=RiscoCredito.COLUNAS)
        df['risco'] = y_raw
        return df

    @staticmethod
    def remover_moderado(df: pd.DataFrame) -> pd.DataFrame:
        """Remove os registros com risco 'moderado'."""
        return df[df['risco'] != 'moderado'].reset_index(drop=True)

    @staticmethod
    def encode_previsores(df: pd.DataFrame) -> np.ndarray:
        """Retorna a matriz de previsores (já codificada numericamente no pkl)."""
        return df[RiscoCredito.COLUNAS].values

    @staticmethod
    def encode_classe(df: pd.DataFrame):
        """Codifica a coluna 'risco' com LabelEncoder e retorna (y_encoded, encoder)."""
        le = LabelEncoder()
        y = le.fit_transform(df['risco'].values)
        return y, le

    @staticmethod
    def novo_cliente(historia: int, divida: int, garantias: int, renda: int) -> np.ndarray:
        """Cria um array para predição de um novo cliente."""
        return np.array([[historia, divida, garantias, renda]])

