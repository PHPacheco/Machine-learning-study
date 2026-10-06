"""Executa sequencialmente o notebook kNN leve sem depender do Jupyter."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "2_kNN" / "knn_com_validacao.ipynb"


def main() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__notebook__"}
    executed = 0
    for index, cell in enumerate(payload["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.strip():
            exec(compile(source, f"{NOTEBOOK}#cell-{index}", "exec"), namespace)
            executed += 1
    result = namespace["result"]
    print(f"{executed} células executadas; métricas: {result.to_dict()}")


if __name__ == "__main__":
    main()
