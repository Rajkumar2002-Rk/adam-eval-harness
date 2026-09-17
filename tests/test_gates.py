"""Regression tests for the gate layer.

Same discipline as ai-org: every gate is locked by a test built from a real
(here, faithfully reproduced) failure, so a future refactor cannot silently
remove a gate's ability to catch it.
"""
from __future__ import annotations

import pytest

from adameval import gates, inject, io, spec as S

DATASETS = [("ADSL", "adsl"), ("ADAE", "adae")]

# Defects that no gate short of a full ground-truth diff can see. Locked in as a
# list because shrinking it is the point of the project - if a future structural
# gate catches one of these, this test fails and the row moves tiers on purpose.
KNOWN_SILENT = {
    "actarm_confusion",
    "trtdurd_off_by_one",
    "aoccifl_first_chronological",
}


@pytest.fixture(scope="module")
def loaded():
    return {ds: (S.load(f"specs/{p}.yaml"), io.read(f"data/adam/{p}.xpt"))
            for ds, p in DATASETS}


@pytest.mark.parametrize("ds,_p", DATASETS)
def test_ground_truth_passes_every_gate(loaded, ds, _p):
    spec, truth = loaded[ds]
    assert gates.run_all(truth, truth, spec) == []


@pytest.mark.parametrize("ds,_p", DATASETS)
def test_prompt_never_leaks_gate_config(loaded, ds, _p):
    spec, _ = loaded[ds]
    for level in S.LEVELS:
        S.assert_no_leakage(S.render_for_prompt(spec, level))


def _cases():
    for ds, _ in DATASETS:
        for name in inject.registry(ds):
            yield ds, name


@pytest.mark.parametrize("ds,defect", list(_cases()))
def test_every_injected_defect_is_caught(loaded, ds, defect):
    spec, truth = loaded[ds]
    found = gates.run_all(inject.apply(ds, defect, truth), truth, spec)
    assert found, f"{defect} was not caught by any gate"


@pytest.mark.parametrize("ds,defect", list(_cases()))
def test_silent_defect_classification_is_stable(loaded, ds, defect):
    spec, truth = loaded[ds]
    found = gates.run_all(inject.apply(ds, defect, truth), truth, spec)
    silent = all(f.layer == "TRUTH" for f in found)
    assert silent == (defect in KNOWN_SILENT), (
        f"{defect} changed tier: silent={silent}. Update KNOWN_SILENT "
        f"deliberately, never to make the suite pass."
    )


def test_provenance_gate_flags_undeclared_derivation(loaded):
    spec, _ = loaded["ADSL"]
    found = gates.provenance({}, spec)
    assert {f.variable for f in found} >= {"AGEGR1", "TRTDURD", "SAFFL"}
    assert all(f.code == "NO_PROVENANCE" for f in found)
