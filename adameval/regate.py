"""Re-run the gate layer over previously generated code.

Every attempt's source is persisted in the run record, so a gate fix can be
applied to past runs without spending a token. This matters because gate bugs
are found by running, and re-paying for a grid each time one surfaces would
make fixing them expensive enough to discourage it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from . import gates, io, runner
from .harness import GROUND_TRUTH, SPECS
from .spec import load


def regate(path: str | Path) -> list[dict[str, Any]]:
    runs = json.loads(Path(path).read_text())
    cache: dict[str, tuple] = {}
    for r in runs:
        ds = r["dataset"]
        if ds not in cache:
            cache[ds] = (load(SPECS[ds]), io.read(GROUND_TRUTH[ds]))
        spec, truth = cache[ds]
        for a in r["attempts"]:
            if not a.get("code"):
                continue
            ex = runner.execute(a["code"], ds)
            a["executed"] = ex.ok
            found = (ex.findings if not ex.ok
                     else gates.run_all(ex.dataset, truth, spec, ex.provenance))
            a["findings"] = [f.to_dict() for f in found]
            if ex.ok:
                r["provenance"] = ex.provenance
        bad = {f["variable"] for f in r["attempts"][-1]["findings"] if f["variable"]}
        r["clean_variables"] = [v for v in r["target_variables"] if v not in bad]
        r["score"] = round(len(r["clean_variables"]) / len(r["target_variables"]), 4)
    return runs
