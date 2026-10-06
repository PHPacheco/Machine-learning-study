"""kNN manual e protocolo sem vazamento para seleção de k."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def euclidean_distance(left: ArrayLike, right: ArrayLike) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        raise ValueError("os vetores precisam ter o mesmo formato")
    return float(np.linalg.norm(left_array - right_array))


def majority_vote(labels: ArrayLike) -> Any:
    values, counts = np.unique(np.asarray(labels).reshape(-1), return_counts=True)
    if len(values) == 0:
        raise ValueError("ao menos um rótulo é necessário")
    # np.unique ordena os rótulos e torna o desempate determinístico.
    return values[np.flatnonzero(counts == counts.max())[0]]


def manual_knn_predict(
    X_train: ArrayLike, y_train: ArrayLike, X_test: ArrayLike, k: int
) -> NDArray[Any]:
    train = np.asarray(X_train, dtype=float)
    test = np.asarray(X_test, dtype=float)
    labels = np.asarray(y_train).reshape(-1)
    if train.ndim != 2 or test.ndim != 2 or train.shape[1] != test.shape[1]:
        raise ValueError("treino e teste devem ser matrizes com o mesmo número de colunas")
    if len(train) != len(labels) or not 1 <= k <= len(train):
        raise ValueError("k inválido ou quantidade de rótulos incompatível")

    predictions = []
    for row in test:
        distances = np.linalg.norm(train - row, axis=1)
        nearest = np.argsort(distances, kind="stable")[:k]
        predictions.append(majority_vote(labels[nearest]))
    return np.asarray(predictions)


def build_knn_search(*, random_state: int = 42, cv_folds: int = 5) -> GridSearchCV:
    pipeline = Pipeline([("scaler", MinMaxScaler()), ("model", KNeighborsClassifier())])
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    return GridSearchCV(
        pipeline,
        {
            "model__n_neighbors": [1, 3, 5, 7, 10, 15, 20],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        scoring={
            "accuracy": "accuracy",
            "macro_f1": "f1_macro",
            "balanced_accuracy": "balanced_accuracy",
        },
        refit="macro_f1",
        cv=cv,
        n_jobs=1,
        error_score="raise",
    )


@dataclass(frozen=True, slots=True)
class KnnResult:
    best_params: dict[str, Any]
    cv_macro_f1_mean: float
    cv_macro_f1_std: float
    test_accuracy: float
    test_macro_f1: float
    test_balanced_accuracy: float
    train_samples: int
    test_samples: int
    classes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_knn(
    X_train: ArrayLike,
    y_train: ArrayLike,
    X_test: ArrayLike,
    y_test: ArrayLike,
    *,
    random_state: int = 42,
    cv_folds: int = 5,
) -> KnnResult:
    train = np.asarray(X_train, dtype=float)
    test = np.asarray(X_test, dtype=float)
    train_labels = np.asarray(y_train).reshape(-1)
    test_labels = np.asarray(y_test).reshape(-1)
    minimum_class_count = int(np.unique(train_labels, return_counts=True)[1].min())
    folds = min(cv_folds, minimum_class_count)
    if folds < 2:
        raise ValueError("cada classe de treino precisa de ao menos duas amostras")

    search = build_knn_search(random_state=random_state, cv_folds=folds)
    max_neighbors = len(train) - int(np.ceil(len(train) / folds))
    search.param_grid["model__n_neighbors"] = [
        value for value in search.param_grid["model__n_neighbors"] if value <= max_neighbors
    ]
    search.fit(train, train_labels)
    predictions = search.predict(test)
    index = search.best_index_
    return KnnResult(
        best_params=dict(search.best_params_),
        cv_macro_f1_mean=float(search.cv_results_["mean_test_macro_f1"][index]),
        cv_macro_f1_std=float(search.cv_results_["std_test_macro_f1"][index]),
        test_accuracy=float(accuracy_score(test_labels, predictions)),
        test_macro_f1=float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
        test_balanced_accuracy=float(balanced_accuracy_score(test_labels, predictions)),
        train_samples=len(train),
        test_samples=len(test),
        classes=len(np.unique(train_labels)),
    )


def load_and_evaluate_mat(
    path: str | Path,
    *,
    train_key: str = "grupoTrain",
    train_label_key: str = "trainRots",
    test_key: str = "grupoTest",
    test_label_key: str = "testRots",
    drop_first_column: bool = False,
) -> KnnResult:
    payload = loadmat(Path(path))
    required = [train_key, train_label_key, test_key, test_label_key]
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"chaves ausentes no MAT: {', '.join(missing)}")
    X_train = np.asarray(payload[train_key], dtype=float)
    X_test = np.asarray(payload[test_key], dtype=float)
    if drop_first_column:
        X_train, X_test = X_train[:, 1:], X_test[:, 1:]
    return evaluate_knn(X_train, payload[train_label_key], X_test, payload[test_label_key])
