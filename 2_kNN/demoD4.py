from pathlib import Path

from ml_studies.knn import load_and_evaluate_mat


def main() -> None:
    result = load_and_evaluate_mat(
        Path(__file__).with_name("grupoDados4.mat"),
        train_key="trainSet",
        train_label_key="trainLabs",
        test_key="testSet",
        test_label_key="testLabs",
        drop_first_column=True,
    )
    print(result.to_dict())


if __name__ == "__main__":
    main()
