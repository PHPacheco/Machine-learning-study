"""Árvore de decisão carregada a partir do CSV, sem artefato pickle."""

from pathlib import Path

from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier

from ml_studies.loaders import load_credit_risk_csv


def main() -> None:
    source = Path(__file__).with_name("dataset_risco_credito.csv")
    X, y = load_credit_risk_csv(source)
    model = DecisionTreeClassifier(criterion="entropy", random_state=42)
    model.fit(X, y)
    predictions = model.predict(X)
    print(
        {
            "balanced_accuracy_treino": balanced_accuracy_score(y, predictions),
            "macro_f1_treino": f1_score(y, predictions, average="macro"),
            "aviso": "métricas no treino são didáticas e não estimam generalização",
        }
    )


if __name__ == "__main__":
    main()
