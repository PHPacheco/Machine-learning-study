"""Código reutilizável dos estudos acadêmicos."""

from .knn import euclidean_distance, majority_vote, manual_knn_predict
from .regression import LinearAnalysis, multiple_linear_regression

__all__ = [
    "LinearAnalysis",
    "euclidean_distance",
    "majority_vote",
    "manual_knn_predict",
    "multiple_linear_regression",
]
