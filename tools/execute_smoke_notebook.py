"""Executa sequencialmente o notebook kNN leve sem depender do Jupyter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NOTEBOOK = ROOT / "2_kNN" / "knn_com_validacao.ipynb"


def main() -> None:
    notebook = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    executed = 0
    for index, cell in enumerate(payload["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.strip():
            exec(compile(source, f"{notebook}#cell-{index}", "exec"), namespace)
            executed += 1
    result = namespace.get("result")
    suffix = f"; métricas: {result.to_dict()}" if result is not None else ""
    print(f"{executed} células executadas{suffix}")


if __name__ == "__main__":
    main()
