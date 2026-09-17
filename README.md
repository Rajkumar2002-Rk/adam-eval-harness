# adam-eval-harness

An eval harness and regression gate for LLM-generated SDTM → ADaM derivations,
run entirely on public data.

**This is not a claim that I can do ADaM derivation.** It is a measurement of
where an LLM breaks when asked to, and a deterministic gate layer that catches
those breaks. The deliverable is the defect table below, not the pipeline.

---

## The finding, in one paragraph

Across 18 runs deriving ADSL and ADAE from public CDISC pilot data, Claude Opus 5
produced a defective dataset 11 times. **Every one of those 11 defective outputs
had zero findings from checks that don't require the answer key** — they were
internally consistent, type-correct, controlled-terminology-clean, and wrong.
A separate six-run arm then fed the ground-truth diffs back to the model, and it
fixed almost all of them: `EOSSTT` went from defective in 6/6 runs to 0/6,
`AOCCIFL` 6/6 → 0/6, `TRTEMFL` 6/6 → 1/6. The model was always capable of the
correct derivation. Nothing in a normal pipeline ever told it that it was wrong.

Detection is the product.

---

## What was built

```
specs/          22 target variables, each with 3 ablation levels + a gate block
adameval/       spec loader, gate layer, sandbox runner, repair loop, CLI
tests/          33 regression tests, 12 synthetic defect injectors
runs/           raw results from every run reported here
notes/          the derivation rules, and how each was verified
```

Pipeline: spec → prompt → model writes Python → execute in a subprocess →
deterministic gates → targeted repair → provenance dump.

**No gate calls an LLM.** The gate layer does not import the model client, and
the dependency is one-directional so the claim is checkable rather than
aspirational.

### The gate layer, in three tiers

The tiers exist so the results can't overclaim. A check that only works because
I read the answer key while writing the spec is not the same as a check a real
conformance tool could run.

| Tier | Needs | Catches |
|---|---|---|
| `STRUCTURAL` | nothing study-specific | types, CDISC controlled terminology, key uniqueness, paired-variable consistency, cardinality invariants |
| `CALIBRATED` | study-specific expected counts, i.e. a QC spec | distribution and missingness deviations |
| `TRUTH` | the full reference dataset | everything else — **these are the silent failures** |

### Ablation

Each variable's spec carries three levels of derivation detail:

- **L0** — the rule spelled out, including study-specific cut points and
  imputation rules
- **L1** — label and source dataset only. *The realistic case*: real sponsor
  specs assume domain knowledge the reader is presumed to have
- **L2** — label only

Everything under a variable's `gate:` key is withheld from the prompt. A test
asserts that gate configuration can never reach the model.

---

## Results

### Scores (fraction of target variables with no finding)

| Dataset | L0 (full rule) | L1 (label + source) | L2 (label only) |
|---|---|---|---|
| ADSL | **1.00, 1.00, 1.00** | 0.90 ×3 | 0.70 ×3 |
| ADAE | **1.00, 1.00, 1.00** | 0.83 ×3 | 0.83, 0.83, 1.00 |

**With a complete specification, the model was perfect — 6/6 runs, 22 variables,
first attempt, no repair needed.** This is not a project about a model being bad
at its task. It is competent at the task and fails on what the spec doesn't say.

The ADSL gradient (1.00 → 0.90 → 0.70) tracks spec completeness cleanly. ADAE
does not, and with three repeats the sample is too small to say whether that is
a real effect or noise. See Limitations.

### Defect table

Defects surviving to the final attempt, across all 18 main-grid runs:

| Freq | Tier | Variable | Rows wrong | Levels | What the model did |
|---|---|---|---|---|---|
| 6/18 | **SILENT** | `ADSL.EOSSTT` | **52** | L1, L2 | Mapped screen failures to `DISCONTINUED` instead of missing |
| 5/18 | **SILENT** | `ADAE.TRTEMFL` | 36 | L1, L2 | Treatment-emergent window wrong |
| 5/18 | **SILENT** | `ADAE.AOCCIFL` | 8 | L1, L2 | Flagged chronologically first event, not first at maximum severity |
| 3/18 | **SILENT** | `ADSL.TRTEDT` | ≤6 | L2 | Invented a last-dose date where none was collected |
| 3/18 | **SILENT** | `ADSL.TRTDURD` | ≤6 | L2 | Cascade from `TRTEDT` |
| 2/18 | PROVENANCE | `ADAE.TRTEMFL`, `AOCCIFL` | — | L2 | Under-declared source columns |

`TRTEMFL` is the one a biometrics programmer should care about most: it is the
flag every safety table filters on, and it was wrong in 5 of 18 runs with no
structural signal whatsoever.

### Three defects worth describing properly

**1. Fabricating a clinical fact nobody asked for.** Two subjects have exactly
one exposure record with a blank `EXENDTC`:

```
01-705-1018   PLACEBO      EXSTDTC 2013-07-05   EXENDTC (blank)
01-705-1382   XANOMELINE   EXSTDTC 2013-05-13   EXENDTC (blank)
```

The reference dataset leaves `TRTEDT` missing — the exposure end is genuinely
unknown. The model substituted the start date and derived `TRTDURD = 1`,
inventing a one-day treatment duration. It affects 2 of 306 subjects, which is
small enough that no summary statistic moves. **I did not predict this one.** It
was found by reading a run, and is now a locked regression test.

The same family appears at scale in ADAE, where 473 of 1,191 events (40%) have
no end date because they were ongoing at the data cut. `AENDTF` is blank for all
1,191 rows — the reference implementation never imputes an end date, because
doing so would fabricate a clinical fact.

**2. A variable name that lies.** `AOCCIFL` reads like "first occurrence flag."
It is actually *first occurrence of the subject's maximum severity*, restricted
to treatment-emergent records. A model that flags the chronologically first event
produces **correct subject counts and the wrong rows** — every incidence table
still totals correctly.

**3. A rule that cannot be inferred from data.** `TRTEMFL` requires
`ASTDT <= TRTEDT + 30 days`. The ground truth constrains that window only to
somewhere in **[8, 36] days** — 14, 28 and 30 all reproduce it exactly. The
window is not recoverable from examples. It can only come from the spec.

### The repair loop

| | Structural feedback only | Full feedback |
|---|---|---|
| ADSL L1 mean | 0.90 | **1.00** |
| ADAE L1 mean | 0.83 | 0.92 |
| Structural defects fed back | 7 | — |
| Structural defects fixed | **6 of 6 (100%)** | — |
| Runs fully resolved | **0 of 6** | 5 of 6 |

By default the repair loop sees only `STRUCTURAL` findings. This is deliberate
and it is the integrity control of the whole experiment: `TRUTH` findings carry
the expected value, so feeding them back would let the model converge by being
handed the answer, and "repair success rate" would measure nothing. It is also
the realistic case — a production pipeline has conformance rules and a QC spec,
not a reference copy of the correct output.

The consequence is the finding. Silent defects are, by construction,
unrepairable: the loop has nothing to say, and the run terminates with a
conformant, wrong dataset. Six runs entered repair with 5–9 findings and came
out with 2–5, every survivor invisible to the feedback channel.

**The repair loop is perfect at what it can see and useless at everything else.
Retry loops fix what your gates detect.**

---

## What this does NOT prove

- **I am not a biometrics programmer.** I learned what ADSL and ADAE are in
  order to build this. A CDISC expert will find things I got wrong.
- **This is public pilot data, not a real study.** CDISCPILOT01 is 306 subjects
  and 1,191 adverse events, and it is clean. Real sponsor data is messier.
- **The "ground truth" is one reference implementation**, not a regulatory gold
  standard. It is `pharmaverseadam`, produced by the `admiral` R package. Where
  the model and admiral disagree, I recorded admiral as correct. That is a
  defensible convention, not a fact.
- **22 of 162 variables.** Chosen because each represents a distinct failure
  mode. The unchosen 140 are mostly verbatim copies from SDTM, which the model
  gets right every time — including them would inflate every score.
- **No CORE conformance rules.** The cdisc-rules-engine integration was scoped
  out to get to measured results inside a weekend. Its absence is why the
  `STRUCTURAL` tier is narrower than a real conformance run would be.
- **Three repeats is not enough to characterise variance.** One arm run scored
  1.00 on a first attempt where three main-grid runs scored 0.83 at the same
  level. Run-to-run variance is real and this design cannot quantify it.
- **The `CALIBRATED` tier is circular by construction.** Its expected counts come
  from the answer key. It is reported separately for exactly that reason.
- **One model, one vendor.** Claude Opus 5, adaptive thinking, default effort.
  Nothing here says anything about any other model.

### Three harness bugs, and what they mean

Three times, a result that looked like a model failure was mine:

1. **Provenance required exact set equality.** The model declared *more* source
   columns than the spec listed — and was right each time. Only under-declaration
   breaks traceability.
2. **Provenance wasn't resolved transitively.** The spec named an intermediate
   (`ADAE.ASTDT`); the model traced to the raw origin (`AE.AESTDTC`). Both are
   valid. This one made every ADAE run score exactly 0.50 and cut the reported
   mean from 0.887 to 0.696.
3. **Provenance defects were unrepairable.** They sat in no allow-list, so a run
   whose only defects were provenance halted after one attempt.

All three were found by asking *why* the model failed, not by a test. **A harness
that overcounts defects is as misleading as one that undercounts**, and the only
reason I caught the second was that ADAE scoring exactly 0.50 in all nine runs
looked too tidy. Undetected harness bugs almost certainly remain.

Generated code is persisted per attempt, so a gate fix can be re-applied to past
runs for free. That is why bug 2 cost nothing to correct.

---

## Reproducing

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Validate the gate layer against 12 synthetic defects — no API key, no cost:

```bash
python -m adameval.cli selftest
```

Run the loop end to end against canned responses — no API key, no cost:

```bash
python -m adameval.cli run --mock --dataset ADSL --level L1 --repeats 1
```

Reproduce the main grid (~$4, needs `ANTHROPIC_API_KEY`):

```bash
python -m adameval.cli run --dataset both --level all --repeats 3 --budget 12.00
```

Re-gate existing runs after a gate change, offline and free:

```bash
python -c "from adameval.regate import regate; regate('runs/grid.json')"
```

Exit codes: `0` pass, `1` below `--fail-under`, `2` usage error, `3` budget
exceeded. `--budget` is a hard ceiling computed from `response.usage`; the grid
aborts and writes partial results rather than overrunning.

Total spend for every number in this README: **$6.83** across 41 model calls.

---

## What I'd do next

1. **Wire in CORE.** The `STRUCTURAL` tier currently approximates what a real
   conformance checker does. Running the actual CDISC rules would either narrow
   the silent set or confirm it — and confirming it is the stronger result.
2. **Widen the grid.** 22 variables, one model, three repeats. Every one of those
   is a limit worth removing, in that order.
3. **Detect the fabrication class directly.** Every silent defect here is the
   model supplying a value where the correct answer was "unknown." A gate that
   flags *any* newly non-missing cell relative to its source would catch them
   without a reference dataset — which would move rows from `TRUTH` to
   `STRUCTURAL`, where they belong.

Item 3 is the one I'd build first. It is the only thing in this repo that would
work on a real study, where no answer key exists.
