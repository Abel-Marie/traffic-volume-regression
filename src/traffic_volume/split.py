"""Chronological splits: earlier years train, the next year validates, the last year tests."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import Config


def chronological_split(clean: pd.DataFrame, cfg: Config) -> dict[str, np.ndarray]:
    """Positions of each split. No shuffling: the future never trains the past."""
    year = clean["date_time"].dt.year.to_numpy()
    s = cfg.split
    return {"train": np.flatnonzero(year <= s.train_until_year),
            "validation": np.flatnonzero(year == s.validation_year),
            "test": np.flatnonzero(year == s.test_year)}


def rows_hash(values) -> str:
    """Fingerprint of a set of hours, used to prove two runs used the same split."""
    joined = ",".join(sorted(str(v) for v in values))
    return hashlib.sha256(joined.encode()).hexdigest()


def split_summary(clean: pd.DataFrame, splits: dict[str, np.ndarray], target: str) -> dict:
    out = {}
    for name, pos in splits.items():
        part = clean.iloc[pos]
        out[name] = {"rows": len(part), "first_hour": str(part["date_time"].min()),
                     "last_hour": str(part["date_time"].max()), "mean_volume": float(part[target].mean())}
    out["test_rows_sha256"] = rows_hash(clean.iloc[splits["test"]]["date_time"].astype(str))
    return out


def time_series_cv(cfg: Config) -> TimeSeriesSplit:
    """Expanding-window cross-validation on the (time-ordered) training rows."""
    return TimeSeriesSplit(n_splits=cfg.tuning.cv_folds)


def fast_sample(clean: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """In fast mode only: every fourth hour, keeping the time order (for smoke runs)."""
    return clean.iloc[::4].reset_index(drop=True) if cfg.is_fast else clean
