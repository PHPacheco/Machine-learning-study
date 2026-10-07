"""Compatibilidade com o material original de kNN.

Novos experimentos devem usar :mod:`ml_studies.knn`, que mantém o scaler
dentro da validação cruzada e reserva o teste para a avaliação final.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from ml_studies.knn import euclidean_distance, manual_knn_predict

dist = euclidean_distance


def meuKnn(dadosTrain, rotuloTrain, dadosTeste, k):
    return manual_knn_predict(dadosTrain, rotuloTrain, dadosTeste, k).reshape(-1, 1)


def ajustar_normalizacao(dados_treino):
    """Ajusta o scaler somente no treino."""

    return MinMaxScaler().fit(np.asarray(dados_treino, dtype=float))


def aplicar_normalizacao(scaler, dados):
    """Transforma validação ou teste com o scaler já ajustado no treino."""

    return scaler.transform(np.asarray(dados, dtype=float))
