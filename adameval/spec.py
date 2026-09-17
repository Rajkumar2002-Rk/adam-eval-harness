"""Spec loading and prompt rendering.

A spec file is two things at once: the instructions handed to the model, and the
configuration for the deterministic gate layer. `render_for_prompt` exists to
keep those apart - everything under a variable's `gate:` key is withheld from
the model, because handing it the controlled terms and expected counts would
mean measuring nothing.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

Level = Literal["L0", "L1", "L2"]
LEVELS: tuple[Level, ...] = ("L0", "L1", "L2")


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    type: str
    source: dict[str, Any] | None
    derivation: dict[str, str | None]
    gate: dict[str, Any]


@dataclass(frozen=True)
class Spec:
    dataset: str
    label: str
    structure: str
    key: list[str]
    expected_rows: int
    sources: dict[str, str]
    variables: list[Variable]
    merge: dict[str, Any] | None = None

    @property
    def names(self) -> list[str]:
        return [v.name for v in self.variables]

    def variable(self, name: str) -> Variable:
        for v in self.variables:
            if v.name == name:
                return v
        raise KeyError(name)


def load(path: str | Path) -> Spec:
    raw = yaml.safe_load(Path(path).read_text())
    return Spec(
        dataset=raw["dataset"],
        label=raw["label"],
        structure=raw["structure"],
        key=raw["key"],
        expected_rows=raw["expected_rows"],
        sources=raw["sources"],
        merge=raw.get("merge"),
        variables=[
            Variable(
                name=v["name"],
                label=v["label"],
                type=v["type"],
                source=v.get("source"),
                derivation=v.get("derivation", {}),
                gate=v.get("gate", {}),
            )
            for v in raw["variables"]
        ],
    )


def render_for_prompt(spec: Spec, level: Level) -> dict[str, Any]:
    """Return the model-visible view of a spec at a given ablation level.

    L0  full derivation text, including study-specific rules
    L1  label and source only - the realistic case, where a sponsor spec
        assumes domain knowledge the reader is presumed to have
    L2  label only - no derivation, no source columns
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level {level!r}, expected one of {LEVELS}")

    out: dict[str, Any] = {
        "dataset": spec.dataset,
        "label": spec.label,
        "structure": spec.structure,
        "key": spec.key,
        "sources": copy.deepcopy(spec.sources),
        "variables": [],
    }
    if spec.merge and level != "L2":
        out["merge"] = copy.deepcopy(spec.merge)

    for v in spec.variables:
        item: dict[str, Any] = {"name": v.name, "label": v.label, "type": v.type}
        if level == "L2":
            if v.source:
                item["source"] = {"dataset": v.source.get("dataset")}
        else:
            if v.source:
                item["source"] = copy.deepcopy(v.source)
            text = v.derivation.get(level)
            if text:
                item["derivation"] = " ".join(str(text).split())
        out["variables"].append(item)
    return out


def assert_no_leakage(rendered: dict[str, Any]) -> None:
    """Fail loudly if gate configuration ever reaches the prompt."""
    blob = yaml.safe_dump(rendered)
    for forbidden in ("gate:", "controlled_terms", "expected_counts", "trap:",
                      "missing_expected", "cross_check", "cardinality", "formula"):
        if forbidden in blob:
            raise AssertionError(f"gate config leaked into prompt: {forbidden!r}")
