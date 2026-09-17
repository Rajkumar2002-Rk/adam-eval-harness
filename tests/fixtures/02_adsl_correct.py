# Mock fixture: the fully correct derivation, used to prove the harness can
# actually reach a clean score rather than being unsatisfiable.
import pandas as pd

PROVENANCE = {
    "USUBJID": ["DM.USUBJID"],
    "TRT01P": ["DM.ARM"], "TRT01A": ["DM.ACTARM"],
    "TRTSDT": ["EX.EXSTDTC"], "TRTEDT": ["EX.EXENDTC"],
    "TRTDURD": ["ADSL.TRTSDT", "ADSL.TRTEDT"], "SAFFL": ["ADSL.TRTSDT"],
    "AGEGR1": ["DM.AGE"], "RACEGR1": ["DM.RACE"], "EOSSTT": ["DS.DSCAT", "DS.DSDECOD"],
}


def derive(sources):
    dm, ex, ds = sources["DM"], sources["EX"], sources["DS"]
    out = dm[["USUBJID", "ARM", "ACTARM", "AGE", "RACE"]].copy()
    out["TRT01P"] = out["ARM"]
    out["TRT01A"] = out["ACTARM"]

    st = ex.groupby("USUBJID")["EXSTDTC"].min()
    en = ex[ex["EXENDTC"] != ""].groupby("USUBJID")["EXENDTC"].max()
    out["TRTSDT"] = out["USUBJID"].map(pd.to_datetime(st, errors="coerce").dt.date)
    out["TRTEDT"] = out["USUBJID"].map(pd.to_datetime(en, errors="coerce").dt.date)
    out["TRTDURD"] = [float((e - s).days + 1) if pd.notna(s) and pd.notna(e) else None
                      for s, e in zip(out["TRTSDT"], out["TRTEDT"])]
    out["SAFFL"] = out["TRTSDT"].notna().map({True: "Y", False: "N"})
    out["AGEGR1"] = out["AGE"].map(lambda a: "18-64" if a <= 64 else ">64")
    out["RACEGR1"] = (out["RACE"] == "WHITE").map({True: "White", False: "Non-white"})

    disp = ds[ds["DSCAT"] == "DISPOSITION EVENT"].set_index("USUBJID")["DSDECOD"]
    code = out["USUBJID"].map(disp)
    out["EOSSTT"] = code.map(lambda c: "COMPLETED" if c == "COMPLETED"
                             else ("" if c == "SCREEN FAILURE" else "DISCONTINUED"))
    return out[["USUBJID", "TRT01P", "TRT01A", "TRTSDT", "TRTEDT",
                "TRTDURD", "SAFFL", "AGEGR1", "RACEGR1", "EOSSTT"]]
