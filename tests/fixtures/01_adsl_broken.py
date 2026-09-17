# Mock fixture: a plausible first attempt that makes three of the predicted
# mistakes - the EXDOSE > 0 filter, prior-knowledge age groups, and TRT01A
# derived from the planned arm.
import pandas as pd

PROVENANCE = {
    "USUBJID": ["DM.USUBJID"],
    "TRT01P": ["DM.ARM"], "TRT01A": ["DM.ARM"],
    "TRTSDT": ["EX.EXSTDTC"], "TRTEDT": ["EX.EXENDTC"],
    "TRTDURD": ["ADSL.TRTSDT", "ADSL.TRTEDT"], "SAFFL": ["ADSL.TRTSDT"],
    "AGEGR1": ["DM.AGE"], "RACEGR1": ["DM.RACE"], "EOSSTT": ["DS.DSCAT", "DS.DSDECOD"],
}


def derive(sources):
    dm, ex, ds = sources["DM"], sources["EX"], sources["DS"]
    out = dm[["USUBJID", "ARM", "AGE", "RACE"]].copy()
    out["TRT01P"] = out["ARM"]
    out["TRT01A"] = out["ARM"]                      # defect: planned, not actual

    dosed = ex[ex["EXDOSE"] > 0]                    # defect: drops placebo
    st = dosed.groupby("USUBJID")["EXSTDTC"].min()
    en = dosed[dosed["EXENDTC"] != ""].groupby("USUBJID")["EXENDTC"].max()
    out["TRTSDT"] = out["USUBJID"].map(pd.to_datetime(st, errors="coerce").dt.date)
    out["TRTEDT"] = out["USUBJID"].map(pd.to_datetime(en, errors="coerce").dt.date)
    out["TRTDURD"] = [(e - s).days + 1 if pd.notna(s) and pd.notna(e) else None
                      for s, e in zip(out["TRTSDT"], out["TRTEDT"])]
    out["SAFFL"] = out["TRTSDT"].notna().map({True: "Y", False: "N"})

    out["AGEGR1"] = pd.cut(out["AGE"], [-1, 64, 80, 200],
                           labels=["<65", "65-80", ">80"]).astype(str)  # defect
    out["RACEGR1"] = (out["RACE"] == "WHITE").map({True: "White", False: "Non-white"})

    disp = ds[ds["DSCAT"] == "DISPOSITION EVENT"].set_index("USUBJID")["DSDECOD"]
    code = out["USUBJID"].map(disp)
    out["EOSSTT"] = code.map(lambda c: "COMPLETED" if c == "COMPLETED"
                             else ("" if c == "SCREEN FAILURE" else "DISCONTINUED"))
    return out[["USUBJID", "TRT01P", "TRT01A", "TRTSDT", "TRTEDT",
                "TRTDURD", "SAFFL", "AGEGR1", "RACEGR1", "EOSSTT"]]
