"""Compara os classificadores usados no trabalho de crédito."""

import pickle
import warnings
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


def carregar_dados(path: Path):
    """Abre o artefato legado mantido no próprio repositório."""

    warnings.warn(
        "Abrindo um arquivo pickle legado. Não use arquivos de origem desconhecida.",
        UserWarning,
        stacklevel=2,
    )
    with path.open("rb") as arquivo:
        return pickle.load(arquivo)


def calcular_metricas(nome, y_real, previsoes):
    return {
        "modelo": nome,
        "acuracia": accuracy_score(y_real, previsoes),
        "macro_f1": f1_score(y_real, previsoes, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_real, previsoes),
    }


def main() -> None:
    caminho = Path(__file__).with_name("dados") / "credit.pkl"
    X_treino, y_treino, X_teste, y_teste = carregar_dados(caminho)[:4]

    modelos = {
        "regressao_logistica": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("modelo", LogisticRegression(random_state=42)),
            ]
        ),
        "naive_bayes": Pipeline(
            [("scaler", StandardScaler()), ("modelo", GaussianNB())]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=42,
        ),
    }

    resultados = []
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)
        resultados.append(calcular_metricas(nome, y_teste, modelo.predict(X_teste)))

    svm = GridSearchCV(
        Pipeline([("scaler", StandardScaler()), ("modelo", SVC())]),
        {"modelo__kernel": ["linear", "rbf", "poly"], "modelo__C": [0.1, 1, 10]},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=1,
    )
    svm.fit(X_treino, y_treino)
    resultados.append(calcular_metricas("svm", y_teste, svm.predict(X_teste)))

    tabela = pd.DataFrame(resultados).set_index("modelo")
    print(tabela.round(4).to_string())
    print("\nParâmetros do SVM escolhidos no treino:", svm.best_params_)


if __name__ == "__main__":
    main()
