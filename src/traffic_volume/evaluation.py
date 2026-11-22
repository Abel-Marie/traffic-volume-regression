"""Regression metrics, block-bootstrap intervals, diagnostic curves, slices and residual analysis."""

from __future__ import annotations

import pickle
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_validate

from .config import Config
from .models import MODEL_NOTES, make_regressor
from .rowlevel import RAINY
from .split import time_series_cv


def regression_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    mse = mean_squared_error(y, pred)
    return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "mae": float(mean_absolute_error(y, pred)),
            "r2": float(r2_score(y, pred))}


def rmse_of(y, pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(pred, float)) ** 2)))


def day_blocks(days) -> tuple[np.ndarray, dict]:
    days = np.asarray(days)
    unique = np.unique(days)
    return unique, {d: np.flatnonzero(days == d) for d in unique}


def block_bootstrap_ci(y, pred, days, n_samples: int, seed: int) -> dict:
    """95% intervals from resampling whole days with replacement.

    Neighbouring hours are strongly correlated, so resampling single hours would
    pretend there is more independent evidence than there is and give intervals
    that are too narrow.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    unique, index = day_blocks(days)
    rng = np.random.default_rng(seed)
    draws: dict[str, list[float]] = {}
    for _ in range(n_samples):
        pick = np.concatenate([index[d] for d in rng.choice(unique, size=len(unique), replace=True)])
        for k, v in regression_metrics(y[pick], pred[pick]).items():
            draws.setdefault(k, []).append(v)
    return {k: {"low": float(np.percentile(v, 2.5)), "high": float(np.percentile(v, 97.5))} for k, v in draws.items()}


def paired_block_bootstrap(y, pred, reference, days, n_samples: int, seed: int) -> dict:
    """95% interval for how much lower the model's RMSE is than a reference model's.

    Both models are scored on the same resampled days, so the interval describes the difference
    itself. Comparing one model's interval with the other's point estimate would not.
    """
    y, pred, reference = np.asarray(y, float), np.asarray(pred, float), np.asarray(reference, float)
    unique, index = day_blocks(days)
    rng = np.random.default_rng(seed)
    cuts, gaps = [], []
    for _ in range(n_samples):
        pick = np.concatenate([index[d] for d in rng.choice(unique, size=len(unique), replace=True)])
        model, ref = rmse_of(y[pick], pred[pick]), rmse_of(y[pick], reference[pick])
        cuts.append(1 - model / ref)
        gaps.append(ref - model)
    point_model, point_ref = rmse_of(y, pred), rmse_of(y, reference)
    return {"rmse_reduction": {"point": 1 - point_model / point_ref, "low": float(np.percentile(cuts, 2.5)),
                               "high": float(np.percentile(cuts, 97.5))},
            "rmse_difference": {"point": point_ref - point_model, "low": float(np.percentile(gaps, 2.5)),
                                "high": float(np.percentile(gaps, 97.5))},
            "share_of_resamples_model_better": float(np.mean(np.array(gaps) > 0))}


def evaluate_regressors(X_train, y_train, X_val, y_val, cats, nums, cfg: Config) -> pd.DataFrame:
    """Time-series CV on the training years, then a full-train fit scored on the validation year."""
    rows = []
    for kind in MODEL_NOTES:
        model = make_regressor(kind, cats, nums, cfg)
        cv = cross_validate(model, X_train, y_train, cv=time_series_cv(cfg), scoring="neg_root_mean_squared_error")
        start = time.time()
        model.fit(X_train, y_train)
        fit_s = time.time() - start
        start = time.time()
        pred = model.predict(X_val)
        per_1k_ms = (time.time() - start) / len(X_val) * 1e6
        val = regression_metrics(y_val, pred)
        rows.append({"model": kind, "cv_rmse_mean": float(-np.mean(cv["test_score"])),
                     "cv_rmse_std": float(np.std(cv["test_score"])), "val_mse": val["mse"], "val_rmse": val["rmse"],
                     "val_mae": val["mae"], "val_r2": val["r2"], "fit_seconds": round(fit_s, 2),
                     "predict_ms_per_1k_rows": round(per_1k_ms, 2),
                     "size_kb": round(len(pickle.dumps(model)) / 1024, 1), "notes": MODEL_NOTES[kind]})
    return pd.DataFrame(rows).sort_values("val_rmse").reset_index(drop=True)


def history_curve(model, X_train, y_train, X_val, y_val, cfg: Config) -> dict:
    """Learning curve for a time series: how much history does the model need?

    The model is refitted on the most recent n training hours (counting back from the end of the
    training years) and scored on those hours and on the whole validation year. Every point is
    scored on the same validation year, so the points are comparable.
    """
    sizes, train, val = [], [], []
    for share in cfg.diagnostics.learning_curve_sizes:
        n = max(int(round(share * len(X_train))), 200)
        Xs, ys = X_train.iloc[-n:], y_train.iloc[-n:]
        fitted = clone(model).fit(Xs, ys)
        sizes.append(n)
        train.append(rmse_of(ys, fitted.predict(Xs)))
        val.append(rmse_of(y_val, fitted.predict(X_val)))
    return {"train_sizes": sizes, "train_rmse": train, "val_rmse": val}


def complexity_curve(kind: str, params: dict, X_train, y_train, X_val, y_val, cats, nums, cfg: Config) -> dict | None:
    """Training and validation error as the model's capacity grows (everything else fixed)."""
    spec = cfg.diagnostics.complexity_curve.get(kind)
    if spec is None:
        return None
    train, val = [], []
    for value in spec["values"]:
        fitted = make_regressor(kind, cats, nums, cfg, **{**params, spec["param"]: value}).fit(X_train, y_train)
        train.append(rmse_of(y_train, fitted.predict(X_train)))
        val.append(rmse_of(y_val, fitted.predict(X_val)))
    chosen = make_regressor(kind, cats, nums, cfg, **params).named_steps["model"].get_params()[spec["param"]]
    return {"param": spec["param"], "values": list(spec["values"]), "train_rmse": train, "val_rmse": val,
            "chosen": chosen}


def fit_verdict(train_rmse: float, val_rmse: float, curve: dict | None) -> dict:
    """Put the under/overfitting evidence into words, from the numbers (nothing is asserted by hand)."""
    gap = val_rmse / train_rmse - 1
    if gap > 0.25:
        over = (f"Validation error is {gap:.0%} above training error: the model overfits, fitting the training "
                "years noticeably better than it predicts a new year.")
    elif gap > 0.05:
        over = (f"Validation error is {gap:.0%} above training error: mild overfitting, the expected price of a "
                "flexible model.")
    elif gap >= -0.05:
        over = (f"Training and validation error differ by only {abs(gap):.1%}: no sign of overfitting.")
    else:
        over = (f"Validation error is {abs(gap):.0%} below training error: no overfitting. If anything the "
                "training years are the harder ones to fit; the table of training error by year shows which.")
    under, best_value = "", None
    if curve:
        best = int(np.argmin(curve["val_rmse"]))
        best_value = curve["values"][best]
        if val_rmse <= curve["val_rmse"][best] * 1.02:
            under = (f" On the complexity curve the validation error is lowest at {curve['param']} = {best_value} "
                     f"({curve['val_rmse'][best]:.0f}) and the chosen setting ({curve['chosen']}) is within 2% of "
                     "it: more capacity would not help, so the model is not underfitting either.")
        else:
            under = (f" On the complexity curve the validation error is lowest at {curve['param']} = {best_value} "
                     f"({curve['val_rmse'][best]:.0f}), better than the chosen setting ({curve['chosen']}): capacity "
                     "is not at its best value.")
    return {"gap": float(gap), "text": over + under, "best_capacity": best_value}


def day_segments(clean_rows: pd.DataFrame) -> np.ndarray:
    """Each hour's kind of time: quiet holiday, weekend, or one of four parts of a working day."""
    t = clean_rows["date_time"]
    hour, weekend = t.dt.hour, t.dt.dayofweek >= 5
    return np.select(
        [clean_rows["quiet_holiday"].to_numpy(), weekend.to_numpy(), hour.isin([6, 7, 8, 9, 15, 16, 17, 18]).to_numpy(),
         hour.between(10, 14).to_numpy(), hour.between(19, 22).to_numpy()],
        ["Quiet holiday", "Weekend", "Weekday rush hours", "Weekday midday", "Weekday evening"], "Weekday night")


def weather_kinds(clean_rows: pd.DataFrame) -> np.ndarray:
    w = clean_rows["weather_main"]
    return np.select([w.isin(RAINY).to_numpy(), (w == "Snow").to_numpy(), w.isin(["Fog", "Mist", "Haze", "Smoke"]).to_numpy()],
                     ["Rain or storm", "Snow", "Low visibility"], "Clear or cloudy")


SEGMENT_ORDER = ["Weekday rush hours", "Weekday midday", "Weekday evening", "Weekday night", "Weekend", "Quiet holiday"]
WEATHER_ORDER = ["Clear or cloudy", "Low visibility", "Rain or storm", "Snow"]


def slice_report(clean_rows: pd.DataFrame, y, preds: dict, reference: str = "baseline", model: str = "champion") -> dict:
    """Error of each model in each kind of hour, each month and each kind of weather.

    ``reduction`` is how much lower the model's RMSE is than the reference's in that slice;
    ``share_of_sq_error`` is the slice's share of the model's total squared error.
    """
    frame = pd.DataFrame({k: np.asarray(v, float) for k, v in preds.items()})
    frame["y"] = np.asarray(y, float)
    t = clean_rows["date_time"]
    groups = {"by_segment": (day_segments(clean_rows), SEGMENT_ORDER),
              "by_month": (t.dt.month.to_numpy(), sorted(t.dt.month.unique())),
              "by_weather": (weather_kinds(clean_rows), WEATHER_ORDER),
              "by_period": (np.where(t.dt.month.to_numpy() <= 9, "January to September", "October to December"),
                            ["January to September", "October to December"])}
    total_sq = float(((frame["y"] - frame[model]) ** 2).sum())
    out = {}
    for name, (labels, order) in groups.items():
        rows = []
        for value in order:
            part = frame[labels == value]
            if part.empty:
                continue
            row = {"group": int(value) if name == "by_month" else value, "hours": len(part),
                   "mean_volume": float(part["y"].mean())}
            for k in preds:
                row[f"{k}_rmse"] = rmse_of(part["y"], part[k])
            row["reduction"] = float(1 - row[f"{model}_rmse"] / row[f"{reference}_rmse"])
            row["share_of_hours"] = len(part) / len(frame)
            row["share_of_sq_error"] = float(((part["y"] - part[model]) ** 2).sum() / total_sq)
            rows.append(row)
        out[name] = rows
    return out


def residual_report(clean_rows: pd.DataFrame, y, pred, target: str, top: int = 10) -> dict:
    """Residuals by hour and by day type, by named day, plus the worst-predicted hours (validation data)."""
    frame = pd.DataFrame({"date_time": clean_rows["date_time"].to_numpy(), "holiday": clean_rows["holiday"].to_numpy(),
                          "actual": np.asarray(y, float), "predicted": np.asarray(pred, float)})
    frame["residual"] = frame["actual"] - frame["predicted"]
    frame["hour"] = frame["date_time"].dt.hour
    frame["day_type"] = np.where(clean_rows["quiet_holiday"].to_numpy(), "quiet holiday",
                                 np.where(frame["holiday"] != "None", "other named day",
                                          np.where(frame["date_time"].dt.dayofweek >= 5, "weekend", "weekday")))
    by_hour = frame.groupby("hour")["residual"].agg(mean="mean", rmse=lambda r: float(np.sqrt(np.mean(r ** 2))))
    by_type = frame.groupby("day_type")["residual"].agg(rows="size", mean="mean",
                                                        rmse=lambda r: float(np.sqrt(np.mean(r ** 2))))
    named = frame[frame["holiday"] != "None"].groupby("holiday")["residual"].agg(
        hours="size", mean="mean", rmse=lambda r: float(np.sqrt(np.mean(r ** 2))))
    worst = frame.reindex(frame["residual"].abs().sort_values(ascending=False).index).head(top)
    worst = worst.assign(date_time=worst["date_time"].astype(str))
    days = frame.assign(day=frame["date_time"].dt.strftime("%Y-%m-%d"), sq=frame["residual"] ** 2)
    worst_days = days.groupby("day").agg(hours=("sq", "size"), rmse=("sq", lambda s: float(np.sqrt(s.mean()))),
                                         mean_residual=("residual", "mean"), holiday=("holiday", "first"))
    worst_days = worst_days.sort_values("rmse", ascending=False).head(top)
    return {"by_hour": by_hour.reset_index().to_dict("records"), "by_day_type": by_type.reset_index().to_dict("records"),
            "by_named_day": named.reset_index().to_dict("records"),
            "worst": worst[["date_time", "holiday", "actual", "predicted", "residual"]].to_dict("records"),
            "worst_days": worst_days.reset_index().to_dict("records"),
            "over_prediction_share_of_large_errors": float((frame.loc[frame["residual"].abs() > 1000, "residual"] < 0).mean())
            if (frame["residual"].abs() > 1000).any() else None,
            "frame": frame}
