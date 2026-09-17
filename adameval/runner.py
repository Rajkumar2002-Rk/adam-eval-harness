"""Runs one derivation attempt: execute generated code, then gate the output."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .findings import Finding

TIMEOUT_S = 180


@dataclass
class Execution:
    ok: bool
    dataset: pd.DataFrame | None
    provenance: dict[str, list[str]]
    findings: list[Finding] = field(default_factory=list)
    stderr: str = ""


def execute(code: str, dataset: str, cwd: str | Path = ".") -> Execution:
    with tempfile.TemporaryDirectory() as tmp:
        mod = Path(tmp) / "derivation.py"
        mod.write_text(code)
        out = Path(tmp) / "result.pkl"
        proc = subprocess.run(
            [sys.executable, "-m", "adameval.sandbox", str(mod), dataset, str(out)],
            cwd=str(cwd), capture_output=True, text=True, timeout=TIMEOUT_S,
        )
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-12:]
            return Execution(False, None, {}, stderr="\n".join(tail), findings=[
                Finding("EXECUTION_FAILED", "EXECUTION", None,
                        f"generated code exited {proc.returncode}",
                        examples=tail[-3:])])
        df = pickle.loads(out.read_bytes())
        prov_file = Path(str(out) + ".prov.json")
        prov = json.loads(prov_file.read_text()) if prov_file.exists() else {}
        return Execution(True, df, prov, stderr=proc.stderr)


def redact_for_repair(findings: list[Finding], allow: set[str]) -> list[dict[str, Any]]:
    """Build repair feedback, withholding anything the model must not be told.

    This is the integrity control of the whole experiment. A TRUTH finding
    carries the expected value; feeding that back would let the model converge by
    being handed the answer key, and the resulting 'repair success rate' would
    measure nothing. By default only STRUCTURAL findings are returned - which is
    also the realistic case, since a production pipeline has conformance rules
    and a QC spec but not a reference copy of the answer.
    """
    out = []
    for f in findings:
        if f.layer == "PROVENANCE":
            # Always actionable and always redacted. The raw message names the
            # spec's source columns, which at L1/L2 would hand back derivation
            # detail the ablation deliberately withheld - so say only that the
            # declaration is incomplete, never which sources are missing.
            out.append({"code": f.code, "variable": f.variable,
                        "problem": "PROVENANCE is incomplete for this variable: it "
                                   "does not list every source column the derivation "
                                   "actually reads."})
            continue
        if f.layer not in allow:
            continue
        item = {"code": f.code, "variable": f.variable, "problem": f.message}
        if f.layer == "STRUCTURAL" and f.examples:
            item["examples"] = f.examples[:3]
        out.append(item)
    return out
