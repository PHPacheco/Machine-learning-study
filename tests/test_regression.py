import numpy as np

from ml_studies.regression import LinearAnalysis, multiple_linear_regression


def test_simple_linear_regression_matches_known_line() -> None:
    analysis = LinearAnalysis()
    intercept, slope = analysis.fit([0, 1, 2, 3], [1, 3, 5, 7])

    assert np.isclose(intercept, 1)
    assert np.isclose(slope, 2)
    assert np.isclose(analysis.correlation([0, 1, 2], [0, 2, 4]), 1)


def test_multiple_regression_handles_full_design_matrix() -> None:
    X = np.asarray([[1, 0, 0], [1, 1, 0], [1, 0, 1], [1, 2, 3]], dtype=float)
    y = 2 + 3 * X[:, 1] - 4 * X[:, 2]

    coefficients = multiple_linear_regression(X, y).reshape(-1)

    np.testing.assert_allclose(coefficients, [2, 3, -4], atol=1e-10)
