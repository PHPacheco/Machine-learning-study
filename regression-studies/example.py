"""Short multiple linear regression example."""

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    file = Path(__file__).with_name("data") / "housing_data.csv"
    data = pd.read_csv(
        file,
        header=None,
        names=["size", "bedrooms", "price"],
    )

    features = data[["size", "bedrooms"]].to_numpy(dtype=float)
    target = data["price"].to_numpy(dtype=float)
    matrix = np.column_stack([np.ones(len(features)), features])
    coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)

    correlations = data.corr(numeric_only=True)["price"].drop("price")
    print("Correlations with price:")
    print(correlations.round(4).to_string())
    print("\nCoefficients [intercept, size, bedrooms]:")
    print(coefficients.round(4).tolist())


if __name__ == "__main__":
    main()
