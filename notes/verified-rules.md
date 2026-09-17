# Verified derivation rules

Every rule in `specs/` was checked against the pharmaverseadam ground truth
before being written down. Nothing here comes from the ADaM IG alone.

## Confirmed exactly

| Rule | Check |
|---|---|
| `TRTDURD = TRTEDT - TRTSDT + 1` | exact, all 252 dosed subjects |
| `SAFFL = "Y"` iff `TRTSDT` non-missing | exact, 306/306 |
| `TRTSDT` = min `EX.EXSTDTC` per subject | exact, 254/254 |
| `ASTDY` = diff + 1 if >= 0 else diff | exact, 1191/1191; no Day 0 present |
| `ADURN = AENDT - ASTDT + 1` | exact, 718/718 |
| `ASTDTF` = `D` iff `AESTDTC` is `YYYY-MM`, `M` iff `YYYY` | exact |
| `ASEV = AESEV`; `ASEVN` = 1/2/3 | exact |
| `AOCCIFL` = first record at max `ASEVN` within `TRTEMFL="Y"`, ties by `AESEQ` | exact, 217/217 |
| `EOSSTT` from `DS.DSDECOD` where `DSCAT = "DISPOSITION EVENT"` | exact, 306/306 |
| `TRT01P = ARM`, `TRT01A = ACTARM` | exact |
| `AGEGR1`: `18-64` = AGE 50..64, `>64` = AGE 65..89 | exact |
| `RACEGR1`: WHITE -> White, all else -> Non-white | exact |

## Corrected during verification

**`TRTEMFL` has an upper bound.** The obvious rule `ASTDT >= TRTSDT` fails on 4
records (subject 01-705-1303, event 37 days after last dose). Adding
`ASTDT <= TRTEDT + 30 days` reproduces the ground truth exactly.

The data constrains the window only to **[8, 36] days** — a 7-day window leaves
4 mismatches, and anything above 36 readmits the excluded event. 14, 28 and 30
are all consistent. **The window is not recoverable from the data; it can only
come from the spec.** That is itself a finding: a model asked to infer this rule
from examples cannot get it right except by luck.

**Start dates are not floored at `TRTSDT`.** The common convention
`max(imputed_date, TRTSDT)` is NOT applied. Year-only dates are historical —
1977, 1982, 1986, 1992, 2001, 2002, 2003, 2007 — imputed to Jan 1 of that year
and left in the past, correctly making those events non-treatment-emergent.

## The trap nobody warned us about

`EX.EXDOSE` is **0 for every PLACEBO record**:

```
{'PLACEBO': [0.0], 'XANOMELINE': [54.0, 81.0]}
```

The standard admiral idiom for first dose filters `EXDOSE > 0`. Applied here it
silently drops all 86 placebo subjects, which cascades into `SAFFL`, `TRTDURD`,
`ASTDY`, `AENDY` and `TRTEMFL` for a third of the study.

This was found by accident: an initial verification filtered `EXDOSE > 0` and
matched only 168 of 254 subjects. Removing the filter matched all 254.

Predicted defect-table rows, unverified until the harness runs:
1. Model fills the 473 missing `AENDT` values (fabricated clinical facts)
2. Model imputes `ASTDT` without setting `ASTDTF` (untraceable derivation)
3. Model filters `EXDOSE > 0` and loses the placebo arm
4. Model uses `<65 / 65-80 / >80` for `AGEGR1` from prior knowledge
5. Model derives `TRT01A` from `ARM` (12 subjects silently wrong)
6. Model emits `"N"` for `TRTEMFL` instead of blank
7. Model flags chronologically-first for `AOCCIFL` instead of first-at-max-severity
8. Model omits the `+1` in `ASTDY` and produces Day 0
