from pathlib import Path

import numpy as np
import pytest

from ml_studies.loaders import load_credit_risk_csv, load_trusted_legacy_pickle


def test_credit_loader_encodes_raw_csv(tmp_path: Path) -> None:
    source = tmp_path / "credit.csv"
    source.write_text(
        "historia,divida,risco\nboa,baixa,baixo\nruim,alta,alto\n",
        encoding="utf-8",
    )

    X, y = load_credit_risk_csv(source)

    assert X.shape == (2, 2)
    np.testing.assert_array_equal(y, ["baixo", "alto"])


def test_pickle_loader_requires_explicit_trust(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="bloqueado"):
        load_trusted_legacy_pickle(tmp_path / "unknown.pkl")
