from pathlib import Path

from tools.check_notebooks import check

ROOT = Path(__file__).resolve().parents[1]


def test_notebooks_are_clean_and_use_python3() -> None:
    paths = [path for path in ROOT.rglob("*.ipynb") if ".portfolio" not in path.parts]
    assert paths
    failures = {str(path): check(path) for path in paths if check(path)}
    assert failures == {}
