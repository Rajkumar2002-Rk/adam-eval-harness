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


def test_mock_run_reaches_clean_score(tmp_path):
    """The harness must be satisfiable: a correct derivation scores 1.0.
    Without this, a low score in a real run is uninterpretable."""
    from pathlib import Path

    from adameval import llm
    from adameval.harness import run_once

    bodies = [f"```python\n{p.read_text()}```"
              for p in sorted(Path("tests/fixtures").glob("*_adsl_*.py"))]
    run = run_once(llm.MockClient(bodies), llm.Ledger(budget_usd=1.0),
                   "ADSL", "L1", max_attempts=4)
    assert run.score == 1.0
    assert len(run.attempts) == 2, "expected one repair round"
    assert run.attempts[0].findings and not run.attempts[-1].findings


def test_repair_feedback_never_leaks_ground_truth():
    """Integrity control: TRUTH findings carry expected values. If those reach
    the model, repair success measures nothing."""
    from adameval import gates, inject, io, runner, spec as S

    spec, truth = S.load("specs/ADSL".replace("ADSL", "adsl") + ".yaml"), io.read("data/adam/adsl.xpt")
    found = gates.run_all(inject.apply("ADSL", "actarm_confusion", truth), truth, spec)
    assert any(f.layer == "TRUTH" for f in found)
    feedback = runner.redact_for_repair(found, {"STRUCTURAL"})
    blob = repr(feedback)
    assert "TRUTH_DIFF" not in blob and "expected" not in blob
