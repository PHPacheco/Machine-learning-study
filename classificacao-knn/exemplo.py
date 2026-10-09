"""Executa os quatro estudos de classificação com kNN."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def avaliar_knn(
    X_treino,
    y_treino,
    X_teste,
    y_teste,
    *,
    sementes: int = 42,
    folds: int = 5,
):
    """Escolhe os parâmetros no treino e avalia uma vez no teste."""

    y_treino = np.asarray(y_treino).reshape(-1)
    y_teste = np.asarray(y_teste).reshape(-1)
    menor_classe = np.unique(y_treino, return_counts=True)[1].min()
    folds = min(folds, int(menor_classe))

    pipeline = Pipeline(
        [("scaler", MinMaxScaler()), ("modelo", KNeighborsClassifier())]
    )
    busca = GridSearchCV(
        pipeline,
        {
            "modelo__n_neighbors": [1, 3, 5, 7, 10, 15, 20],
            "modelo__weights": ["uniform", "distance"],
            "modelo__p": [1, 2],
        },
        scoring="f1_macro",
        refit=True,
        cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=sementes),
        n_jobs=1,
    )
    busca.fit(X_treino, y_treino)
    previsoes = busca.predict(X_teste)

    return {
        "melhores_parametros": busca.best_params_,
        "macro_f1_validacao": busca.best_score_,
        "acuracia_teste": accuracy_score(y_teste, previsoes),
        "macro_f1_teste": f1_score(
            y_teste, previsoes, average="macro", zero_division=0
        ),
        "balanced_accuracy_teste": balanced_accuracy_score(y_teste, previsoes),
    }


def carregar_e_avaliar(caminho: Path, *, d4: bool = False):
    dados = loadmat(caminho)
    if d4:
        X_treino = dados["trainSet"][:, 1:]
        y_treino = dados["trainLabs"]
        X_teste = dados["testSet"][:, 1:]
        y_teste = dados["testLabs"]
    else:
        X_treino = dados["grupoTrain"]
        y_treino = dados["trainRots"]
        X_teste = dados["grupoTest"]
        y_teste = dados["testRots"]
    return avaliar_knn(X_treino, y_treino, X_teste, y_teste)


def main() -> None:
    pasta = Path(__file__).with_name("dados")
    for indice in range(1, 5):
        resultado = carregar_e_avaliar(
            pasta / f"grupoDados{indice}.mat",
            d4=indice == 4,
        )
        print(f"D{indice}:", resultado)


if __name__ == "__main__":
    main()
