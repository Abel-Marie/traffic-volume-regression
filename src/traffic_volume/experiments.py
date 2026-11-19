"""Lag features: how much would recent traffic help, at which forecast horizon, and why the deployed
model does without it."""

from __future__ import annotations

import pandas as pd

from .config import Config
from .evaluation import regression_metrics
from .features import split_feature_types
from .models import make_regressor

LAGS = {"lag_1h": 1, "lag_24h": 24, "lag_168h": 168}

# A count from k hours ago is only known when the forecast is made less than k hours ahead.
LAG_HORIZONS = [
    ("Calendar and weather only (the deployed inputs)", [], "any horizon"),
    ("+ same hour last week", ["lag_168h"], "up to 7 days ahead"),
    ("+ same hour yesterday", ["lag_24h", "lag_168h"], "up to 1 day ahead"),
    ("+ previous hour", ["lag_1h", "lag_24h", "lag_168h"], "next hour only; needs a live count feed"),
]
LAG_MODEL = "hgb"


def lag_frame(clean: pd.DataFrame, target: str) -> pd.DataFrame:
    """Volume 1 hour, 1 day and 1 week earlier, aligned on the true hourly clock.

    The series has gaps, so shifting rows would pair hours that are not
    neighbours; reindexing to every hour first leaves a gap as a missing value.
    """
    series = clean.set_index("date_time")[target]
    grid = series.reindex(pd.date_range(series.index.min(), series.index.max(), freq="h"))
    lags = pd.DataFrame({name: grid.shift(k) for name, k in LAGS.items()})
    return lags.reindex(clean["date_time"]).set_index(clean.index)


def lag_experiment(X: pd.DataFrame, y: pd.Series, lags: pd.DataFrame, splits: dict, cfg: Config) -> pd.DataFrame:
    """Validation error with the lags that would be known at each forecast horizon.

    Every row uses the same model family with default settings (HistGradientBoosting, which accepts
    the missing values that gaps in the series leave in the lag columns), fitted on the training years.
    """
    rows = []
    known = lags.iloc[splits["train"]].notna().any()  # a lag with no values at all cannot be used (fast mode)
    for label, names, horizon in LAG_HORIZONS:
        names = [n for n in names if known[n]]
        frame = X.join(lags[names]) if names else X
        cats, nums = split_feature_types(list(frame.columns))
        model = make_regressor(LAG_MODEL, cats, nums, cfg)
        model.fit(frame.iloc[splits["train"]], y.iloc[splits["train"]])
        m = regression_metrics(y.iloc[splits["validation"]], model.predict(frame.iloc[splits["validation"]]))
        rows.append({"features": label, "lags": ", ".join(names) or "none", "usable": horizon, **m})
    avail = lags.iloc[splits["validation"]].notna().mean()
    out = pd.DataFrame(rows)
    out.attrs["lag_available_share"] = {k: float(v) for k, v in avail.items()}
    return out
