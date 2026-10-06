"""Validação leve de higiene e compatibilidade dos notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("%pip install", "!pip install", "python 2", "c:\\users\\")


def check(path: Path) -> list[str]:
    errors: list[str] = []
    notebook = json.loads(path.read_text(encoding="utf-8"))
    kernel = notebook.get("metadata", {}).get("kernelspec", {}).get("name")
    if kernel != "python3":
        errors.append(f"kernel inesperado: {kernel!r}")
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        source = "".join(cell.get("source", [])).lower()
        marker = next((value for value in FORBIDDEN if value in source), None)
        if marker:
            errors.append(f"célula {index}: contém {marker!r}")
        if cell.get("cell_type") == "code" and cell.get("outputs"):
            errors.append(f"célula {index}: contém output")
    return errors


def main() -> None:
    failures = []
    paths = [path for path in sorted(ROOT.rglob("*.ipynb")) if ".portfolio" not in path.parts]
    for path in paths:
        for error in check(path):
            failures.append(f"{path.relative_to(ROOT)}: {error}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"{len(paths)} notebooks canônicos validados")


if __name__ == "__main__":
    main()
