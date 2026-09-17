"""Synthetic defect injection.

Every injector reproduces one predicted LLM failure from notes/verified-rules.md
by corrupting the ground truth in exactly that way. This validates the gate layer
before a single token is spent: a gate that cannot catch a defect we deliberately
planted will not catch it in the wild either.

Each injector takes the ground truth and returns a corrupted copy.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

ADSL_DEFECTS: dict[str, str] = {}
ADAE_DEFECTS: dict[str, str] = {}


def _adsl(desc):
    def deco(fn):
        ADSL_DEFECTS[fn.__name__] = desc
        return fn
    return deco


def _adae(desc):
    def deco(fn):
        ADAE_DEFECTS[fn.__name__] = desc
        return fn
    return deco


# ---------------------------------------------------------------- ADSL

@_adsl("filters EX.EXDOSE > 0, dropping every placebo subject")
def placebo_dose_filter(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    m = d.TRT01A.str.contains("Placebo", na=False)
    d.loc[m, ["TRTSDT", "TRTEDT", "TRTDURD"]] = None
    d.loc[m, "SAFFL"] = "N"
    return d


@_adsl("uses the published CDISC pilot age groups (<65 / 65-80 / >80)")
def agegr1_prior_knowledge(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["AGEGR1"] = pd.cut(d.AGE, [-1, 64, 80, 200],
                         labels=["<65", "65-80", ">80"]).astype(str)
    return d


@_adsl("derives actual treatment from the planned arm")
def actarm_confusion(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["TRT01A"] = d["TRT01P"]
    return d


@_adsl("omits the inclusive +1 in treatment duration")
def trtdurd_off_by_one(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["TRTDURD"] = d.TRTDURD - 1
    return d


@_adsl("maps screen failures to DISCONTINUED instead of missing")
def eosstt_screenfail(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.loc[d.EOSSTT == "", "EOSSTT"] = "DISCONTINUED"
    return d


@_adsl("fills a blank last-dose date with the start date, inventing a 1-day exposure")
def trtedt_single_dose_fabrication(df: pd.DataFrame) -> pd.DataFrame:
    """Captured from a live Opus 5 run, not predicted. Two subjects have exactly
    one EX record with a blank EXENDTC; ground truth leaves TRTEDT missing and
    the model substituted the start date, yielding TRTDURD = 1."""
    d = df.copy()
    m = d.TRTEDT.isna() & d.TRTSDT.notna()
    d.loc[m, "TRTEDT"] = d.loc[m, "TRTSDT"]
    d.loc[m, "TRTDURD"] = 1.0
    return d


# ---------------------------------------------------------------- ADAE

@_adae("fills every missing end date with the last dose date")
def fabricate_end_dates(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    m = d.AENDT.isna()
    d.loc[m, "AENDT"] = d.loc[m, "TRTEDT"]
    filled = d.loc[m]
    d.loc[m, "ADURN"] = (filled.AENDT - filled.ASTDT).apply(
        lambda x: x.days + 1 if pd.notna(x) else None)
    d.loc[m, "ADURU"] = "days"
    delta = (d.loc[m, "AENDT"] - d.loc[m, "TRTSDT"]).apply(
        lambda x: x.days if pd.notna(x) else None)
    d.loc[m, "AENDY"] = delta.apply(lambda v: v + 1 if pd.notna(v) and v >= 0 else v)
    return d


@_adae("imputes start dates without setting the imputation flag")
def silent_imputation(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ASTDTF"] = ""
    return d


@_adae('emits "N" for non-treatment-emergent instead of blank')
def trtemfl_uses_n(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d.loc[d.TRTEMFL == "", "TRTEMFL"] = "N"
    return d


@_adae("subtracts dates without the +1, creating a Day 0")
def astdy_no_plus_one(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ASTDY"] = (d.ASTDT - d.TRTSDT).apply(lambda x: x.days if pd.notna(x) else None)
    return d


@_adae("flags the chronologically first event instead of first-at-max-severity")
def aoccifl_first_chronological(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["AOCCIFL"] = ""
    te = d[d.TRTEMFL == "Y"].sort_values(["USUBJID", "ASTDT", "AESEQ"])
    first = te.groupby("USUBJID").head(1).index
    d.loc[first, "AOCCIFL"] = "Y"
    return d


@_adae("floors imputed start dates at the treatment start date")
def impute_floored_at_trtsdt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    m = d.ASTDTF != ""
    d.loc[m, "ASTDT"] = d.loc[m].apply(
        lambda r: max(r.ASTDT, r.TRTSDT) if pd.notna(r.TRTSDT) else r.ASTDT, axis=1)
    delta = (d.loc[m, "ASTDT"] - d.loc[m, "TRTSDT"]).apply(
        lambda x: x.days if pd.notna(x) else None)
    d.loc[m, "ASTDY"] = delta.apply(lambda v: v + 1 if pd.notna(v) and v >= 0 else v)
    d.loc[m, "TRTEMFL"] = d.loc[m].apply(
        lambda r: "Y" if pd.notna(r.TRTSDT) and r.ASTDT >= r.TRTSDT
        and r.ASTDT <= r.TRTEDT + dt.timedelta(days=30) else "", axis=1)
    return d


def registry(dataset: str) -> dict[str, str]:
    return {"ADSL": ADSL_DEFECTS, "ADAE": ADAE_DEFECTS}[dataset.upper()]


def apply(dataset: str, name: str, df: pd.DataFrame) -> pd.DataFrame:
    fn = globals()[name]
    assert name in registry(dataset), f"{name} is not a {dataset} defect"
    return fn(df)
