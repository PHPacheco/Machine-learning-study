"""Run the four kNN classification studies."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def evaluate_knn(
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    seed: int = 42,
    folds: int = 5,
):
    """Select parameters on training data and evaluate once on test data."""

    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    smallest_class = np.unique(y_train, return_counts=True)[1].min()
    folds = min(folds, int(smallest_class))

    pipeline = Pipeline(
        [("scaler", MinMaxScaler()), ("model", KNeighborsClassifier())]
    )
    search = GridSearchCV(
        pipeline,
        {
            "model__n_neighbors": [1, 3, 5, 7, 10, 15, 20],
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2],
        },
        scoring="f1_macro",
        refit=True,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed),
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    predictions = search.predict(X_test)

    return {
        "best_parameters": search.best_params_,
        "validation_macro_f1": search.best_score_,
        "test_accuracy": accuracy_score(y_test, predictions),
        "test_macro_f1": f1_score(
            y_test, predictions, average="macro", zero_division=0
        ),
        "test_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
    }


def load_and_evaluate(path: Path, *, d4: bool = False):
    data = loadmat(path)
    if d4:
        X_train = data["trainSet"][:, 1:]
        y_train = data["trainLabs"]
        X_test = data["testSet"][:, 1:]
        y_test = data["testLabs"]
    else:
        X_train = data["grupoTrain"]
        y_train = data["trainRots"]
        X_test = data["grupoTest"]
        y_test = data["testRots"]
    return evaluate_knn(X_train, y_train, X_test, y_test)


def main() -> None:
    data_dir = Path(__file__).with_name("data")
    for index in range(1, 5):
        result = load_and_evaluate(
            data_dir / f"dataset_group_{index}.mat",
            d4=index == 4,
        )
        print(f"D{index}:", result)


if __name__ == "__main__":
    main()
