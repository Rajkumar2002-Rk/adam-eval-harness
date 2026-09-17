"""The deterministic gate layer.

Hard rule for this whole module: no gate may call an LLM. Every check here is
either arithmetic, a set membership test, or a comparison against the
pharmaverseadam ground truth.
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from .findings import Finding
from .spec import Spec, Variable

MAX_EXAMPLES = 5


def _isna(s: pd.Series) -> pd.Series:
    return s.isna()


def _nonnull(s: pd.Series) -> pd.Series:
    return s[s.notna()]


def _type_ok(series: pd.Series, declared: str) -> tuple[bool, str]:
    vals = _nonnull(series)
    if vals.empty:
        return True, ""
    sample = vals.iloc[0]
    if declared == "date":
        if isinstance(sample, dt.datetime):
            return False, "datetime where a date is expected"
        if isinstance(sample, dt.date):
            return True, ""
        return False, f"{type(sample).__name__} where a date is expected"
    if declared == "numeric":
        return (pd.api.types.is_numeric_dtype(vals),
                "" if pd.api.types.is_numeric_dtype(vals)
                else f"{type(sample).__name__} where a number is expected")
    if declared == "character":
        return (isinstance(sample, str), "" if isinstance(sample, str)
                else f"{type(sample).__name__} where a string is expected")
    return True, ""


def structural(candidate: pd.DataFrame, spec: Spec) -> list[Finding]:
    """Checks that need no ground truth: presence, grain, types, terminology,
    declared counts, cardinality, and paired variables."""
    out: list[Finding] = []

    missing = [n for n in spec.names if n not in candidate.columns]
    for n in missing:
        out.append(Finding("MISSING_VAR", "STRUCTURAL", n,
                           "required variable absent from candidate dataset"))

    if len(candidate) != spec.expected_rows:
        out.append(Finding("ROW_COUNT", "CALIBRATED", None,
                           f"expected {spec.expected_rows} rows, got {len(candidate)}",
                           n_rows=abs(len(candidate) - spec.expected_rows)))

    if all(k in candidate.columns for k in spec.key):
        dup = candidate.duplicated(subset=spec.key, keep=False)
        if dup.any():
            out.append(Finding("KEY_DUP", "STRUCTURAL", "+".join(spec.key),
                               f"key {spec.key} is not unique",
                               n_rows=int(dup.sum()),
                               examples=candidate.loc[dup, spec.key]
                               .head(MAX_EXAMPLES).astype(str).to_dict("records")))

    for v in spec.variables:
        if v.name not in candidate.columns:
            continue
        col = candidate[v.name]
        g: dict[str, Any] = v.gate

        ok, why = _type_ok(col, v.type)
        if not ok:
            out.append(Finding("TYPE_MISMATCH", "STRUCTURAL", v.name, why))

        if (terms := g.get("controlled_terms")) is not None:
            allowed = set(terms)
            seen = set(_nonnull(col).unique())
            if bad := seen - allowed:
                n = int(col.isin(list(bad)).sum())
                out.append(Finding("CT_VIOLATION", "STRUCTURAL", v.name,
                                   f"values outside controlled terminology: "
                                   f"{sorted(map(repr, bad))[:MAX_EXAMPLES]}",
                                   n_rows=n, examples=sorted(map(str, bad))[:MAX_EXAMPLES]))

        if (bad_vals := g.get("forbidden_values")) is not None:
            hit = col.isin(bad_vals)
            if hit.any():
                out.append(Finding("FORBIDDEN_VALUE", "STRUCTURAL", v.name,
                                   f"forbidden value(s) {bad_vals} present",
                                   n_rows=int(hit.sum())))

        if (exp := g.get("missing_expected")) is not None:
            got = int(_isna(col).sum())
            if got != exp:
                out.append(Finding("MISSING_COUNT", "CALIBRATED", v.name,
                                   f"expected {exp} missing values, got {got}",
                                   n_rows=abs(got - exp)))

        if (counts := g.get("expected_counts")) is not None:
            got = col.value_counts(dropna=False).to_dict()
            got = {("" if pd.isna(k) else k): int(n) for k, n in got.items()}
            want = {k: int(n) for k, n in counts.items()}
            if got != want:
                out.append(Finding("VALUE_COUNTS", "CALIBRATED", v.name,
                                   f"value distribution differs: expected {want}, got {got}"))

        if (card := g.get("cardinality")) is not None:
            out.extend(_cardinality(candidate, v, card))

        if (pair := g.get("paired_with")) is not None:
            out.extend(_paired(candidate, v, pair))

    return out


def _cardinality(df: pd.DataFrame, v: Variable, card: dict[str, Any]) -> list[Finding]:
    per = card["per"]
    if not all(c in df.columns for c in per + [v.name]):
        return []
    scope = df
    if (within := card.get("within")):
        wv = within["variable"]
        if wv not in df.columns:
            return []
        scope = df[df[wv] == within["equals"]]
    eligible = set(scope[per[0]].unique()) if len(per) == 1 else None
    flagged = scope[scope[v.name] == card["flag_value"]]
    counts = flagged.groupby(per).size()
    want = int(card.get("count", 1))
    wrong = counts[counts != want]
    out: list[Finding] = []
    if not wrong.empty:
        out.append(Finding("CARDINALITY", "STRUCTURAL", v.name,
                           f"expected exactly {want} {card['flag_value']!r} per {per}, "
                           f"{len(wrong)} group(s) differ",
                           n_rows=len(wrong),
                           examples=[str(i) for i in wrong.index[:MAX_EXAMPLES]]))
    if eligible is not None:
        unflagged = eligible - set(counts.index)
        if unflagged:
            out.append(Finding("CARDINALITY", "STRUCTURAL", v.name,
                               f"{len(unflagged)} eligible group(s) have no "
                               f"{card['flag_value']!r} at all",
                               n_rows=len(unflagged),
                               examples=sorted(map(str, unflagged))[:MAX_EXAMPLES]))
    return out


def _paired(df: pd.DataFrame, v: Variable, pair: dict[str, Any]) -> list[Finding]:
    other, mapping = pair["variable"], pair["mapping"]
    if other not in df.columns or v.name not in df.columns:
        return []
    expected = df[other].map(mapping)
    bad = (expected != df[v.name]) & expected.notna()
    if not bad.any():
        return []
    ex = df.loc[bad, [other, v.name]].head(MAX_EXAMPLES).astype(str).to_dict("records")
    return [Finding("PAIR_MISMATCH", "STRUCTURAL", v.name,
                    f"{v.name} inconsistent with {other}",
                    n_rows=int(bad.sum()), examples=ex)]


def truth_diff(candidate: pd.DataFrame, truth: pd.DataFrame, spec: Spec) -> list[Finding]:
    """Cell-level comparison against pharmaverseadam. Anything caught only here
    is a silent failure: the candidate was internally consistent and still wrong."""
    out: list[Finding] = []
    key = spec.key
    if not all(k in candidate.columns for k in key):
        return [Finding("KEY_ABSENT", "TRUTH", "+".join(key),
                        "cannot diff: key columns missing from candidate")]

    c = candidate.drop_duplicates(subset=key).set_index(key).sort_index()
    t = truth.drop_duplicates(subset=key).set_index(key).sort_index()

    only_c, only_t = c.index.difference(t.index), t.index.difference(c.index)
    if len(only_c):
        out.append(Finding("EXTRA_KEYS", "TRUTH", None,
                           f"{len(only_c)} key(s) present in candidate but not ground truth",
                           n_rows=len(only_c), examples=[str(i) for i in only_c[:MAX_EXAMPLES]]))
    if len(only_t):
        out.append(Finding("MISSING_KEYS", "TRUTH", None,
                           f"{len(only_t)} key(s) present in ground truth but not candidate",
                           n_rows=len(only_t), examples=[str(i) for i in only_t[:MAX_EXAMPLES]]))

    shared = c.index.intersection(t.index)
    for v in spec.variables:
        if v.name in key or v.name not in c.columns or v.name not in t.columns:
            continue
        cv, tv = c.loc[shared, v.name], t.loc[shared, v.name]
        both_na = cv.isna() & tv.isna()
        differs = ~both_na & ((cv != tv) | (cv.isna() != tv.isna()))
        if not differs.any():
            continue
        idx = differs[differs].index[:MAX_EXAMPLES]
        ex = [{"key": str(i), "expected": str(tv.loc[i]), "got": str(cv.loc[i])} for i in idx]
        out.append(Finding("TRUTH_DIFF", "TRUTH", v.name,
                           f"{int(differs.sum())} of {len(shared)} values differ from ground truth",
                           n_rows=int(differs.sum()), examples=ex))
    return out


def provenance(declared: dict[str, list[str]], spec: Spec) -> list[Finding]:
    """Every derived variable must declare where it came from, and the claim must
    match the spec. An undeclared derivation is an untraceable one."""
    out: list[Finding] = []
    for v in spec.variables:
        want = v.gate.get("provenance")
        if not want:
            continue
        got = declared.get(v.name)
        if not got:
            out.append(Finding("NO_PROVENANCE", "PROVENANCE", v.name,
                               "no source declared for a derived variable"))
        elif set(got) != set(want):
            out.append(Finding("PROVENANCE_MISMATCH", "PROVENANCE", v.name,
                               f"declared {sorted(got)}, spec says {sorted(want)}"))
    return out


def run_all(candidate: pd.DataFrame, truth: pd.DataFrame, spec: Spec,
            declared_provenance: dict[str, list[str]] | None = None) -> list[Finding]:
    f = structural(candidate, spec) + truth_diff(candidate, truth, spec)
    if declared_provenance is not None:
        f += provenance(declared_provenance, spec)
    return f
