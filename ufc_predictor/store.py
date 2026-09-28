"""Tiny CSV-backed storage layer (no parquet/pyarrow dependency)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from . import config

DATE_COLS = {"event_date", "date", "created_at", "graded_at"}


def read(path: Path, **kw) -> pd.DataFrame:
    if not Path(path).exists() or Path(path).stat().st_size == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False, **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    for c in df.columns:
        if c in DATE_COLS or c.endswith("_dob"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def write(df: pd.DataFrame, path: Path) -> None:
    config.ensure_dirs()
    tmp = Path(str(path) + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def upsert(new: pd.DataFrame, path: Path, key: str) -> pd.DataFrame:
    """Append rows, replacing any existing row with the same key."""
    old = read(path)
    if old.empty or key not in old.columns:
        out = new.copy()
    elif new.empty:
        out = old
    else:
        out = pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)
    write(out, path)
    return out


def data_status() -> dict:
    f = read(config.FIGHTS_CSV)
    u = read(config.UPCOMING_CSV)
    fi = read(config.FIGHTERS_CSV)
    return {
        "fights": len(f),
        "fighters": len(fi),
        "upcoming": len(u),
        "first_date": None if f.empty else f["event_date"].min(),
        "last_date": None if f.empty else f["event_date"].max(),
        "has_odds_history": bool((not f.empty) and "f_1_odds" in f and f["f_1_odds"].notna().any()),
        "model_trained": config.MODEL_FILE.exists(),
    }
