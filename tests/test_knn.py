import numpy as np

from ml_studies.knn import (
    build_knn_search,
    euclidean_distance,
    majority_vote,
    manual_knn_predict,
)


def test_distance_and_deterministic_vote() -> None:
    assert np.isclose(euclidean_distance([0, 0], [3, 4]), 5)
    assert majority_vote([2, 1, 2, 1]) == 1


def test_manual_knn_predicts_nearest_cluster() -> None:
    X_train = np.asarray([[0, 0], [0, 1], [10, 10], [10, 11]])
    y_train = np.asarray([0, 0, 1, 1])

    predictions = manual_knn_predict(X_train, y_train, [[0.2, 0.1], [9.8, 10.2]], k=1)

    np.testing.assert_array_equal(predictions, [0, 1])


def test_scaler_is_inside_grid_search_pipeline() -> None:
    search = build_knn_search(random_state=4, cv_folds=2)

    assert list(search.estimator.named_steps) == ["scaler", "model"]
