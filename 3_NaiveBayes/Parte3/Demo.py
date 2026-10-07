"""Comparação executável de classificadores, sem backend gráfico obrigatório."""

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

from ml_studies.loaders import load_trusted_legacy_pickle


def _metrics(name, y_true, predictions):
    return {
        "modelo": name,
        "acuracia": accuracy_score(y_true, predictions),
        "macro_f1": f1_score(y_true, predictions, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
    }


def main() -> None:
    legacy = Path(__file__).parents[1] / "Parte2" / "credit.pkl"
    X_train, y_train, X_test, y_test = load_trusted_legacy_pickle(legacy, trusted=True)[:4]
    models = {
        "regressao_logistica": Pipeline(
            [("scaler", StandardScaler()), ("model", LogisticRegression(random_state=42))]
        ),
        "naive_bayes": Pipeline([("scaler", StandardScaler()), ("model", GaussianNB())]),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
    }
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        rows.append(_metrics(name, y_test, model.predict(X_test)))

    svm = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("model", SVC())]),
        {"model__kernel": ["linear", "rbf", "poly"], "model__C": [0.1, 1, 10]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
    )
    svm.fit(X_train, y_train)
    rows.append(_metrics("svm", y_test, svm.predict(X_test)))
    print(pd.DataFrame(rows).set_index("modelo").round(4).to_string())
    print("SVM selecionado somente no treino:", svm.best_params_)


if __name__ == "__main__":
    main()
