"""Executes model-written derivation code in a subprocess.

The contract with the model is deliberately narrow: define `derive(sources)`
returning a DataFrame, and a module-level PROVENANCE dict mapping each derived
variable to the source columns it came from. Anything else is a contract
violation the runner records as an EXECUTION defect.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

SOURCES = {
    "ADSL": {"DM": "data/sdtm/dm.xpt", "EX": "data/sdtm/ex.xpt", "DS": "data/sdtm/ds.xpt"},
    "ADAE": {"AE": "data/sdtm/ae.xpt", "ADSL": "data/adam/adsl.xpt"},
}


def main() -> int:
    module_path, dataset, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    import pandas as pd  # noqa: F401  (available to generated code)
    import pyreadstat

    sources = {}
    for name, path in SOURCES[dataset].items():
        df, _ = pyreadstat.read_xport(path)
        sources[name] = df

    ns: dict = {}
    code = Path(module_path).read_text()
    exec(compile(code, "<derivation>", "exec"), ns)  # noqa: S102 - that is the job

    if "derive" not in ns:
        print("CONTRACT: no derive() defined", file=sys.stderr)
        return 3
    result = ns["derive"](sources)
    if not hasattr(result, "columns"):
        print(f"CONTRACT: derive() returned {type(result).__name__}, not a DataFrame",
              file=sys.stderr)
        return 4

    Path(out_path).write_bytes(pickle.dumps(result))
    Path(out_path + ".prov.json").write_text(json.dumps(ns.get("PROVENANCE", {})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
