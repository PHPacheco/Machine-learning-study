"""Demonstra o pré-processamento reproduzível a partir do CSV bruto."""

from pathlib import Path

from ml_studies.loaders import load_credit_risk_csv


def main() -> None:
    source = Path(__file__).with_name("dataset_risco_credito.csv")
    X, y = load_credit_risk_csv(source)
    print(f"Fonte: {source}")
    print(f"Previsores: {X.shape} | classes: {sorted(set(y))}")


if __name__ == "__main__":
    main()
