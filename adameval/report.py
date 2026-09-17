"""Run records and scoring."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .findings import Finding
from .llm import Usage


@dataclass
class Attempt:
    n: int
    findings: list[Finding] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    executed: bool = False
    code: str = ""


@dataclass
class Run:
    dataset: str
    level: str
    repeat: int
    model: str
    attempts: list[Attempt] = field(default_factory=list)
    provenance: dict[str, list[str]] = field(default_factory=dict)
    target_variables: list[str] = field(default_factory=list)

    @property
    def final(self) -> Attempt:
        return self.attempts[-1]

    @property
    def clean_variables(self) -> list[str]:
        bad = {f.variable for f in self.final.findings if f.variable}
        return [v for v in self.target_variables if v not in bad]

    @property
    def score(self) -> float:
        """Fraction of target variables with no finding of any tier."""
        if not self.target_variables:
            return 0.0
        return len(self.clean_variables) / len(self.target_variables)

    @property
    def usage(self) -> Usage:
        u = Usage()
        for a in self.attempts:
            u = u + a.usage
        return u

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset, "level": self.level, "repeat": self.repeat,
            "model": self.model, "score": round(self.score, 4),
            "n_attempts": len(self.attempts),
            "target_variables": self.target_variables,
            "clean_variables": self.clean_variables,
            "provenance": self.provenance,
            "cost_usd": round(self.usage.cost(self.model), 4),
            "attempts": [
                {"n": a.n, "executed": a.executed,
                 "findings": [f.to_dict() for f in a.findings],
                 "usage": dataclasses.asdict(a.usage)}
                for a in self.attempts
            ],
        }


def write(runs: list[Run], path: str | Path) -> None:
    Path(path).write_text(json.dumps([r.to_dict() for r in runs], indent=2, default=str))
