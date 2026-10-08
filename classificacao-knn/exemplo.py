"""Executa os quatro estudos de kNN com seleção feita somente no treino."""

from pathlib import Path

from ml_studies.knn import load_and_evaluate_mat


def main() -> None:
    dados = Path(__file__).with_name("dados")
    for indice in range(1, 5):
        parametros = {}
        if indice == 4:
            parametros = {
                "train_key": "trainSet",
                "train_label_key": "trainLabs",
                "test_key": "testSet",
                "test_label_key": "testLabs",
                "drop_first_column": True,
            }
        resultado = load_and_evaluate_mat(dados / f"grupoDados{indice}.mat", **parametros)
        print(f"D{indice}:", resultado.to_dict())


if __name__ == "__main__":
    main()
