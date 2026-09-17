"""A finding is one defect, attributed to the gate that caught it.

The `layer` field is the whole point of the project. Three tiers, in order of
how much prior knowledge they need:

  STRUCTURAL  catchable with no study-specific answer key - types, CDISC
              controlled terminology, key uniqueness, internal consistency
              between paired variables, cardinality invariants. This is roughly
              what an off-the-shelf conformance checker can do.
  CALIBRATED  needs study-specific expected counts, i.e. a QC spec or someone
              who already knows the answer. Honest about its own dependency:
              these checks look strong here only because we read the ground
              truth while writing the spec.
  TRUTH       needs the full reference dataset. Anything caught only here is a
              silent failure - internally valid, conformant, and wrong. These
              are the rows worth writing about.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

Layer = Literal["STRUCTURAL", "CALIBRATED", "TRUTH", "PROVENANCE", "EXECUTION"]


@dataclass
class Finding:
    code: str
    layer: Layer
    variable: str | None
    message: str
    n_rows: int = 0
    examples: list[Any] = field(default_factory=list)

    @property
    def silent(self) -> bool:
        """True if only the ground-truth diff could have caught this."""
        return self.layer == "TRUTH"

    @property
    def needs_answer_key(self) -> bool:
        """True if catching this required knowing the study-specific answer."""
        return self.layer in ("CALIBRATED", "TRUTH")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["silent"] = self.silent
        d["needs_answer_key"] = self.needs_answer_key
        return d

    def __str__(self) -> str:
        where = f" [{self.variable}]" if self.variable else ""
        n = f" ({self.n_rows} rows)" if self.n_rows else ""
        return f"{self.layer}/{self.code}{where}{n}: {self.message}"
