"""Steps shared by the pipeline stages and the notebook.

The pipeline wraps each step with disk input/output; the notebook calls the
same steps in narrative order, so both produce the same results.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.base import clone

from .cleaning import parse_calendar, typical_weather, with_typical_weather
from .config import Config
from .evaluation import (
    block_bootstrap_ci,
    complexity_curve,
    fit_verdict,
    history_curve,
    paired_block_bootstrap,
    regression_metrics,
    residual_report,
    slice_report,
)
from .feature_study import CALENDAR_VARIABLES, family_model, model_features
from .features import build_feature_frame, split_feature_types
from .models import TREE_KINDS, make_regressor
from .rowlevel import RAW_VARIABLES, features_using_only
from .tuning import best_lgbm

INPUT_SETS = {"all": ("all inputs", list(RAW_VARIABLES)), "calendar": ("calendar only", CALENDAR_VARIABLES)}


def build_frames(clean: pd.DataFrame, splits: dict, cfg: Config) -> dict:
    """Raw variables, features and targets for every split, plus the cleaned rows."""
    raw = parse_calendar(clean)
    X = build_feature_frame(raw, cfg.features.sources)
    y = clean[cfg.target].astype(float)
    cats, nums = split_feature_types(list(X.columns))
    out = {"sources": cfg.features.sources, "cats": cats, "nums": nums, "X": X, "y": y, "raw": raw, "splits": splits}
    for name, pos in splits.items():
        out[f"X_{name}"], out[f"y_{name}"] = X.iloc[pos], y.iloc[pos]
        out[f"raw_{name}"], out[f"clean_{name}"] = raw.iloc[pos], clean.iloc[pos]
    trval = np.concatenate([splits["train"], splits["validation"]])
    out["X_trval"], out["y_trval"], out["clean_trval"] = X.iloc[trval], y.iloc[trval], clean.iloc[trval]
    return out


def finalist_label(kind: str, tuned: bool, inputs: str) -> str:
    return f"{kind}, {'tuned' if tuned else 'default settings'}, {INPUT_SETS[inputs][0]}"


def finalists(tuning: dict) -> list[dict]:
    """The models that compete on the validation year.

    Each tree family appears four times: with the settings its search chose and with its default
    settings, and each of those on all inputs and on the calendar inputs alone (no weather).
    Cross-validation nominates settings; the validation year decides.
    """
    searches = {"rf": tuning["rf_random"], "hgb": tuning["hgb_random"], "lgbm": best_lgbm(tuning)}
    out = [{"label": "ridge, tuned, all inputs", "kind": "ridge", "search": tuning["ridge_grid"], "inputs": "all",
            "tuned": True}]
    for inputs in INPUT_SETS:
        for kind, search in searches.items():
            for tuned in (True, False):
                out.append({"label": finalist_label(kind, tuned, inputs), "kind": kind,
                            "search": search if tuned else None, "inputs": inputs, "tuned": tuned})
    return out


def input_columns(inputs: str, f: dict) -> tuple[list[str], list[str]]:
    """The categorical and numeric feature columns of an input set. A pipeline built on them ignores
    every other column of the feature frame."""
    allowed = set(features_using_only(INPUT_SETS[inputs][1]))
    return [c for c in f["cats"] if c in allowed], [c for c in f["nums"] if c in allowed]


def fit_tuned(tuning: dict, f: dict, cfg: Config) -> dict:
    """Fit every finalist on the training years and score it on the validation year."""
    out = {}
    for item in finalists(tuning):
        search = item["search"]
        params = search["best_params"] if search else {}
        cats, nums = input_columns(item["inputs"], f)
        model = make_regressor(item["kind"], cats, nums, cfg, **params).fit(f["X_train"], f["y_train"])
        out[item["label"]] = {
            "model": model, "kind": item["kind"], "tuned": item["tuned"], "inputs": item["inputs"],
            "variables": len(INPUT_SETS[item["inputs"]][1]), "params": params,
            "strategy": f"{search['strategy']} search" if search else "default settings",
            "cv_rmse": search["best_rmse"] if search else None,
            **{f"val_{k}": v for k, v in regression_metrics(f["y_validation"], model.predict(f["X_validation"])).items()},
            "size_kb": round(len(pickle.dumps(model)) / 1024, 1)}
    return out


def choose_champion(fitted: dict, cfg: Config) -> dict:
    """Lowest validation RMSE among the tree models (the ones the service can serve). Models within the
    tie tolerance count as equal, and the simplest of them wins: fewest input variables, then smallest file."""
    trees = {k: v for k, v in fitted.items() if v["kind"] in TREE_KINDS}
    best = min(v["val_rmse"] for v in trees.values())
    tied = {k: v for k, v in trees.items() if v["val_rmse"] - best <= cfg.selection.tie_tolerance_rmse}
    label = min(tied, key=lambda k: (tied[k]["variables"], tied[k]["size_kb"]))
    win = fitted[label]
    other = "calendar" if win["inputs"] == "all" else "all"
    return {"name": win["kind"], "label": label, "params": win["params"], "strategy": win["strategy"],
            "inputs": win["inputs"], "variables": INPUT_SETS[win["inputs"]][1],
            "counterpart_label": finalist_label(win["kind"], win["tuned"], other), "counterpart_inputs": other,
            "all_inputs_label": finalist_label(win["kind"], win["tuned"], "all"),
            "val_mse": win["val_mse"], "val_rmse": win["val_rmse"], "val_mae": win["val_mae"], "val_r2": win["val_r2"],
            "size_kb": win["size_kb"], "best_val_rmse": best, "best_label": min(trees, key=lambda k: trees[k]["val_rmse"]),
            "tied_models": sorted(tied),
            "rule": f"lowest validation RMSE; models within {cfg.selection.tie_tolerance_rmse} vehicles/hour count as "
                    "equal and the simplest of them wins (fewest input variables, then smallest file)",
            "comparison": [{"model": k, **{kk: vv for kk, vv in v.items() if kk != "model"}}
                           for k, v in fitted.items()]}


def score_without_weather(model, fitted_on: pd.DataFrame, raw_vars: pd.DataFrame, y, cfg: Config) -> dict:
    """The model's error when the weather is not known and each month's typical weather is assumed in its
    place. The typical values come from the rows the model was fitted on."""
    typical = typical_weather(fitted_on)
    X = build_feature_frame(with_typical_weather(raw_vars, typical), cfg.features.sources)
    return regression_metrics(y, model.predict(X))


def diagnose(model, champion: dict, f: dict, cfg: Config) -> dict:
    """Under/overfitting evidence and error analysis for the champion, all on training and validation data."""
    pred = model.predict(f["X_validation"])
    train = regression_metrics(f["y_train"], model.predict(f["X_train"]))
    validation = regression_metrics(f["y_validation"], pred)
    cats, nums = input_columns(champion["inputs"], f)
    curve = complexity_curve(champion["name"], champion["params"], f["X_train"], f["y_train"], f["X_validation"],
                             f["y_validation"], cats, nums, cfg)
    references = {kind: make_regressor(kind, f["cats"], f["nums"], cfg).fit(f["X_train"], f["y_train"])
                  .predict(f["X_validation"]) for kind in ("baseline", "baseline_holiday")}
    years = f["clean_train"]["date_time"].dt.year.to_numpy()
    fitted_values = model.predict(f["X_train"])
    by_year = [{"year": int(y), "hours": int((years == y).sum()),
                **regression_metrics(f["y_train"].to_numpy()[years == y], fitted_values[years == y])}
               for y in sorted(set(years))]
    return {"train": train, "validation": validation, "val_pred": pred, "train_by_year": by_year,
            "validation_typical_weather": score_without_weather(model, f["clean_train"], f["raw_validation"],
                                                                f["y_validation"], cfg),
            "verdict": fit_verdict(train["rmse"], validation["rmse"], curve),
            "learning_curve": history_curve(model, f["X_train"], f["y_train"], f["X_validation"], f["y_validation"], cfg),
            "complexity_curve": curve,
            "residuals": residual_report(f["clean_validation"], f["y_validation"], pred, cfg.target),
            "slices": slice_report(f["clean_validation"], f["y_validation"], {**references, "champion": pred})}


def test_pass_models(champion: dict, tuning: dict, study: dict, f: dict, cfg: Config) -> dict:
    """The pre-declared test models as unfitted pipelines with the columns each one uses."""
    params, kind = champion["params"], champion["name"]
    every = list(f["X"].columns)
    specs = {
        "baseline": (make_regressor("baseline", f["cats"], f["nums"], cfg), every),
        "baseline_holiday": (make_regressor("baseline_holiday", f["cats"], f["nums"], cfg), every),
        "ridge": (make_regressor("ridge", f["cats"], f["nums"], cfg, **tuning["ridge_grid"]["best_params"]), every),
        "univariate": (family_model("tree", study["top1"], params, cfg, kind), model_features("tree", study["top1"])),
        "bivariate": (family_model("tree", study["top2"], params, cfg, kind), model_features("tree", study["top2"])),
        # the champion's family and settings on the other input set (with weather if the champion has none)
        "alternative_inputs": (make_regressor(kind, *input_columns(champion["counterpart_inputs"], f), cfg, **params), every),
        "champion": (make_regressor(kind, *input_columns(champion["inputs"], f), cfg, **params), every),
    }
    if list(specs) != list(cfg.test_pass_models):
        raise ValueError(f"test pass must score exactly {cfg.test_pass_models}")
    return specs


def columns_used(model, fallback: int) -> int:
    """How many feature columns a fitted pipeline actually reads (its column selector may take a subset)."""
    prep = model.named_steps.get("prep")
    if prep is not None:
        return sum(len(cols) for _, _, cols in prep.transformers)
    return {"HourOfWeekBaseline": 2, "HolidayAwareBaseline": 3}.get(type(model.named_steps["model"]).__name__, fallback)


def final_test(specs: dict, f: dict, cfg: Config) -> dict:
    """The only function that reads the test year. Each pre-declared model is refitted on the
    training and validation years (the most recent data matters for a time series) and scored once."""
    results, fitted = {}, {}
    for name, (pipe, cols) in specs.items():
        model = clone(pipe).fit(f["X_trval"][cols], f["y_trval"])
        pred = model.predict(f["X_test"][cols])
        results[name] = {"metrics": regression_metrics(f["y_test"], pred), "pred": pred,
                         "features": columns_used(model, len(cols))}
        fitted[name] = model
    days = f["clean_test"]["date_time"].dt.date.astype(str).to_numpy()
    champ, base, table = results["champion"], results["baseline"], results["baseline_holiday"]
    results["alternative_inputs"]["label"] = "all inputs (with weather)" if "weather_group" not in \
        fitted["champion"].named_steps["prep"].get_feature_names_out() else "calendar only (no weather)"
    n, seed = cfg.evaluation.bootstrap_samples, cfg.seed
    champ["ci"] = block_bootstrap_ci(f["y_test"], champ["pred"], days, n, seed)
    champ["typical_weather_metrics"] = score_without_weather(fitted["champion"], f["clean_trval"], f["raw_test"],
                                                             f["y_test"], cfg)
    base["ci"] = block_bootstrap_ci(f["y_test"], base["pred"], days, n, seed)
    paired = {ref: paired_block_bootstrap(f["y_test"], champ["pred"], results[ref]["pred"], days, n, seed)
              for ref in ("baseline", "baseline_holiday", "alternative_inputs")}
    preds = {k: results[k]["pred"] for k in ("baseline", "baseline_holiday", "alternative_inputs", "champion")}
    frame = pd.DataFrame({"date_time": f["clean_test"]["date_time"].to_numpy(), "actual": f["y_test"].to_numpy(), **preds})
    b, c = base["metrics"], champ["metrics"]
    months = sorted(f["clean_test"]["date_time"].dt.month.unique().tolist())
    return {"models": results, "fitted": fitted, "frame": frame, "paired": paired,
            "slices": slice_report(f["clean_test"], f["y_test"], preds),
            "test_months": months,
            "success": {
                "criterion": f"test R2 >= {cfg.success.r2} and RMSE below the hour-of-week baseline",
                "r2_target_met": bool(c["r2"] >= cfg.success.r2),
                "baseline_r2": b["r2"], "baseline_also_meets_r2_target": bool(b["r2"] >= cfg.success.r2),
                "beats_baseline": bool(c["rmse"] < b["rmse"]),
                "rmse_reduction_vs_baseline": float(1 - c["rmse"] / b["rmse"]),
                "mse_reduction_vs_baseline": float(1 - c["mse"] / b["mse"]),
                "rmse_reduction_vs_holiday_table": float(1 - c["rmse"] / table["metrics"]["rmse"]),
                "reduction_ci_excludes_zero": bool(paired["baseline"]["rmse_reduction"]["low"] > 0)}}


def drop_unserialisable(obj):
    """Drop fitted models, prediction arrays and residual frames before writing JSON."""
    skip = {"pred", "val_pred", "frame", "fitted", "curves"}
    if isinstance(obj, dict):
        return {k: drop_unserialisable(v) for k, v in obj.items() if k not in skip and not hasattr(v, "fit")}
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict("records") if obj.index.name else obj.to_dict("records")
    if isinstance(obj, list):
        return [drop_unserialisable(v) for v in obj]
    return obj


def results_report(parts: dict, cfg: Config) -> dict:
    """The single results file. Docs and slides read their numbers from here."""
    return drop_unserialisable({
        "project": cfg.project, "fast_mode": cfg.is_fast, "seed": cfg.seed,
        "data": {"audit": parts["audit"], "cleaning": parts["cleaning_log"], "hours_after_cleaning": parts["hours"],
                 "split": parts["split_summary"], "holiday_evidence": parts["holiday_evidence"]},
        "features": {"sources": cfg.features.sources, "categorical": parts["cats"], "numeric": parts["nums"]},
        "candidates": parts["candidates"], "lag_experiment": parts["lags"],
        "tuning": {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in parts["tuning"].items()},
        "champion": parts["champion"], "diagnostics": parts["diagnostics"],
        "explain": {"permutation_top10": parts["permutation"].head(10), "shap_top10": parts["shap"].head(10)},
        "feature_study": parts["study"], "test": parts["test"], "first_evaluation": cfg.first_evaluation,
        "deployment": parts.get("deployment"),
    })
