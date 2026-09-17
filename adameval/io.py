"""Dataset loading. XPT is SAS Transport v5 - the reason every column name is
capped at 8 characters."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyreadstat


def read(path: str | Path) -> pd.DataFrame:
    df, _ = pyreadstat.read_xport(str(path))
    return df


def read_labels(path: str | Path) -> dict[str, str]:
    _, meta = pyreadstat.read_xport(str(path))
    return dict(meta.column_names_to_labels)
