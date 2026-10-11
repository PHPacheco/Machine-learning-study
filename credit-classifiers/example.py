"""Compare the classifiers used in the credit study."""

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_data(path: Path):
    """Open the legacy artifact tracked in this repository."""

    with path.open("rb") as file:
        return pickle.load(file)


def calculate_metrics(name, y_true, predictions):
    return {
        "model": name,
        "accuracy": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
    }


def main() -> None:
    path = Path(__file__).with_name("data") / "credit.pkl"
    X_train, y_train, X_test, y_test = load_data(path)[:4]

    models = {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(random_state=42)),
            ]
        ),
        "naive_bayes": Pipeline(
            [("scaler", StandardScaler()), ("model", GaussianNB())]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        results.append(calculate_metrics(name, y_test, model.predict(X_test)))

    svm = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
        {"model__kernel": ["linear", "rbf", "poly"], "model__C": [0.1, 1, 10]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
    )
    svm.fit(X_train, y_train)
    results.append(calculate_metrics("svm", y_test, svm.predict(X_test)))

    table = pd.DataFrame(results).set_index("model")
    print(table.round(4).to_string())
    print("\nSVM parameters selected on the training data:", svm.best_params_)


if __name__ == "__main__":
    main()
