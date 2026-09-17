"""Prompt construction.

Stable content (task contract, SDTM schemas) goes in `system` with cache
breakpoints; the volatile part (the spec at a given ablation level, and any
repair feedback) goes in `messages`. That ordering is what makes the cache hit.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from . import io
from .sandbox import SOURCES
from .spec import Level, Spec, assert_no_leakage, render_for_prompt

TASK = """\
You are writing a Python function that derives an ADaM analysis dataset from SDTM
source datasets, for a clinical trial submission.

Contract - your output is executed exactly as written:

1. Return one ```python code block and nothing else that matters.
2. Define `derive(sources)` taking a dict of pandas DataFrames keyed by SDTM
   domain name, returning a single pandas DataFrame.
3. Define a module-level dict `PROVENANCE` mapping EVERY variable you produce -
   including ones copied unchanged - to the list of source columns it came from, as "DATASET.COLUMN" strings.
   Example: PROVENANCE = {"TRTDURD": ["ADSL.TRTSDT", "ADSL.TRTEDT"]}
4. Produce exactly the variables named in the specification, at the stated
   structure and key. Do not add extra variables.
5. You may import pandas, numpy, and the standard library. Nothing else.
6. Date-typed variables must be datetime.date objects, not strings and not
   datetimes. Character variables must be Python strings.

Derive only what the specification asks for. Where the specification is silent,
make the most defensible choice for a regulated submission and state it in a
comment beginning `# ASSUMPTION:`.
"""


def _schema_block() -> str:
    parts = []
    for path in sorted({p for d in SOURCES.values() for p in d.values()}):
        name = Path(path).stem.upper()
        df = io.read(path)
        labels = io.read_labels(path)
        cols = "\n".join(f"  {c} ({df[c].dtype}) - {labels.get(c, '')}".rstrip(" -")
                         for c in df.columns)
        sample = df.head(3).astype(str).to_csv(index=False)
        parts.append(f"### {name} ({len(df)} rows)\n{cols}\n\nFirst 3 rows:\n{sample}")
    return "## Available source datasets\n\n" + "\n\n".join(parts)


def system_blocks() -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": TASK, "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": _schema_block(), "cache_control": {"type": "ephemeral"}},
    ]


def initial_message(spec: Spec, level: Level) -> dict[str, Any]:
    rendered = render_for_prompt(spec, level)
    assert_no_leakage(rendered)
    body = yaml.safe_dump(rendered, sort_keys=False, width=100)
    return {"role": "user", "content":
            f"Derive the {spec.dataset} dataset from this specification.\n\n"
            f"```yaml\n{body}```"}


def repair_message(feedback: list[dict[str, Any]], attempt: int) -> dict[str, Any]:
    lines = [f"- {f['code']} on {f['variable'] or '(dataset)'}: {f['problem']}"
             + (f"\n  examples: {f['examples']}" if f.get("examples") else "")
             for f in feedback]
    return {"role": "user", "content":
            f"The dataset your code produced failed validation (attempt {attempt}). "
            f"Automated checks reported:\n\n" + "\n".join(lines) +
            "\n\nReturn the corrected full function in a single ```python block. "
            "Do not explain - return the code."}
