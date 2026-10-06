from pathlib import Path

from ml_studies.knn import load_and_evaluate_mat


def main() -> None:
    result = load_and_evaluate_mat(Path(__file__).with_name("grupoDados3.mat"))
    print(result.to_dict())


if __name__ == "__main__":
    main()
