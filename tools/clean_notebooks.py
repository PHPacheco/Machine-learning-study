"""Limpa saídas e normaliza metadados dos notebooks canônicos."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean_notebook(path: Path) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    clean_cells = []
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "code" and any(
            marker in source.lower() for marker in ("%pip install", "!pip install")
        ):
            continue
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        clean_cells.append(cell)

    notebook["cells"] = clean_cells
    metadata = notebook.setdefault("metadata", {})
    metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    metadata["language_info"] = {"name": "python", "version": "3.11"}
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    for path in sorted(ROOT.rglob("*.ipynb")):
        if ".portfolio" not in path.parts:
            clean_notebook(path)
            print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
