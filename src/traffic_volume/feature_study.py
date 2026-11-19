"""Feature importance by raw variable, and models built from one, two or all variables.

The instructor's extra requirement for the regression problem. A "variable" is
one raw input (hour, weekday, temperature, ...). Each derived feature belongs to
the variables it is computed from (rowlevel.FEATURE_INPUTS), so a model "on
hour" never sees a feature that also uses the weekday.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from .config import Config
from .evaluation import regression_metrics, rmse_of, weather_kinds
from .features import build_feature_frame, split_feature_types
from .models import make_regressor
from .rowlevel import FEATURE_INPUTS, RAW_VARIABLES, features_using_only

DISCRETE_VARIABLES = {"hour", "weekday", "month", "is_holiday", "weather_main"}
CALENDAR_VARIABLES = ["hour", "weekday", "month", "is_holiday"]
INPUT_GROUPS = [("Hour and weekday", ["hour", "weekday"]),
                ("+ month", ["hour", "weekday", "month"]),
                ("+ holiday (all calendar inputs)", CALENDAR_VARIABLES),
                ("+ weather (all inputs)", list(RAW_VARIABLES))]

# The single untransformed column for each variable, used by the plain linear model.
RAW_FEATURE_OF = {"hour": "hour", "weekday": "weekday", "month": "month", "is_holiday": "is_holiday",
                  "temp": "temp_c", "rain_1h": "rain_1h", "snow_1h": "snow_1h", "clouds_all": "clouds_all",
                  "weather_main": "weather_group"}


def correlation_ratio(categories, values) -> float:
    """Eta: the share of the target's spread explained by group means (0 to 1)."""
    frame = pd.DataFrame({"c": np.asarray(categories), "v": np.asarray(values, float)})
    means = frame.groupby("c")["v"].transform("mean")
    total = ((frame["v"] - frame["v"].mean()) ** 2).sum()
    return float(np.sqrt(((means - frame["v"].mean()) ** 2).sum() / total)) if total else 0.0


def association(raw_train: pd.DataFrame, y_train) -> dict[str, float]:
    """Correlation ratio for discrete variables, absolute Pearson correlation for continuous ones."""
    out = {}
    for var in RAW_VARIABLES:
        col = raw_train[var]
        out[var] = (correlation_ratio(col, y_train) if var in DISCRETE_VARIABLES
                    else float(abs(np.corrcoef(col.astype(float), np.asarray(y_train, float))[0, 1])))
    return out


def mutual_information(raw_train: pd.DataFrame, y_train, seed: int) -> dict[str, float]:
    X = raw_train[RAW_VARIABLES].copy()
    X["weather_main"] = X["weather_main"].astype("category").cat.codes
    mi = mutual_info_regression(X.astype(float), np.asarray(y_train, float),
                                discrete_features=[v in DISCRETE_VARIABLES for v in RAW_VARIABLES], random_state=seed)
    return dict(zip(RAW_VARIABLES, map(float, mi)))


def grouped_permutation(model, raw_val: pd.DataFrame, y_val, cfg: Config) -> dict[str, float]:
    """Increase in validation RMSE when one raw variable is shuffled and every feature is recomputed."""
    rng = np.random.default_rng(cfg.seed)
    sources = cfg.features.sources
    base = regression_metrics(y_val, model.predict(build_feature_frame(raw_val, sources)))["rmse"]
    out = {}
    for var in RAW_VARIABLES:
        scores = []
        for _ in range(cfg.explain.permutation_repeats):
            shuffled = raw_val.copy()
            shuffled[var] = rng.permutation(shuffled[var].to_numpy())
            scores.append(regression_metrics(y_val, model.predict(build_feature_frame(shuffled, sources)))["rmse"])
        out[var] = float(np.mean(scores) - base)
    return out


def share_by_variable(per_feature: dict[str, float]) -> dict[str, float]:
    """Split each feature's value equally among the variables it uses."""
    out = dict.fromkeys(RAW_VARIABLES, 0.0)
    for feat, value in per_feature.items():
        inputs = FEATURE_INPUTS[feat]
        for var in inputs:
            out[var] += float(value) / len(inputs)
    return out


def split_gain(est) -> np.ndarray:
    """Total error reduction of the splits on each input column.

    LightGBM reports it directly and scikit-learn's forests and single trees expose the equivalent
    ``feature_importances_``. HistGradientBoosting has no such attribute, so the gains are summed
    from the split nodes of its trees.
    """
    if hasattr(est, "booster_"):
        return est.booster_.feature_importance(importance_type="gain")
    if hasattr(est, "feature_importances_"):
        return est.feature_importances_
    gain = np.zeros(est.n_features_in_)
    for stage in est._predictors:
        for tree in stage:
            nodes = tree.nodes
            split = nodes["is_leaf"] == 0
            np.add.at(gain, nodes["feature_idx"][split], nodes["gain"][split])
    return gain


def tree_gain(model) -> dict[str, float]:
    """The model's own split-gain importance, shared out among the raw variables."""
    names = list(model.named_steps["prep"].get_feature_names_out())
    return share_by_variable(dict(zip(names, split_gain(model.named_steps["model"]))))


def grouped_shap(values) -> dict[str, float]:
    """Mean absolute SHAP per variable; a shared feature's SHAP is split among its variables."""
    out = dict.fromkeys(RAW_VARIABLES, 0.0)
    per_row = {var: np.zeros(values.values.shape[0]) for var in RAW_VARIABLES}
    for j, feat in enumerate(values.feature_names):
        inputs = FEATURE_INPUTS[feat]
        for var in inputs:
            per_row[var] += values.values[:, j] / len(inputs)
    for var in RAW_VARIABLES:
        out[var] = float(np.abs(per_row[var]).mean())
    return out


def rank_variables(raw_train, y_train, raw_val, y_val, model, shap_values, cfg: Config) -> pd.DataFrame:
    """Five rankings (association, mutual information, permutation, tree importance, SHAP) and their mean rank."""
    table = pd.DataFrame({
        "association": association(raw_train, y_train),
        "mutual_information": mutual_information(raw_train, y_train, cfg.seed),
        "permutation_rmse_increase": grouped_permutation(model, raw_val, y_val, cfg),
        "tree_gain": tree_gain(model),
        "shap": grouped_shap(shap_values),
    })
    ranks = table.rank(ascending=False, method="min").add_suffix("_rank")
    out = table.join(ranks)
    out["mean_rank"] = ranks.mean(axis=1)
    return out.sort_values(["mean_rank", "permutation_rmse_increase"], ascending=[True, False]).rename_axis("variable")


def vif_table(X_train: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors of the numeric features (multicollinearity)."""
    num = X_train[split_feature_types(list(X_train.columns))[1]].astype(float)
    num = num.loc[:, num.std() > 0]
    arr = np.column_stack([np.ones(len(num)), num.to_numpy()])
    vif = [variance_inflation_factor(arr, i + 1) for i in range(num.shape[1])]
    return pd.DataFrame({"feature": num.columns, "vif": vif}).sort_values("vif", ascending=False).reset_index(drop=True)


def family_model(family: str, variables: list[str], params: dict, cfg: Config, tree_kind: str = "lgbm") -> Pipeline:
    """A model restricted to the features of ``variables``.

    linear_raw: ordinary least squares on the untransformed columns only.
    linear: ordinary least squares on all of the variables' features. spline: periodic
    splines of the hour (one variable, several columns) plus the other features
    linearly. tree: the champion's tree model (``tree_kind``) with its settings.
    """
    feats = model_features(family, variables)
    cats, nums = split_feature_types(feats)
    if family == "linear_raw":
        return make_regressor("linear", cats, nums, cfg)
    if family == "tree":
        return make_regressor(tree_kind, cats, nums, cfg, **params)
    if family == "linear":
        return make_regressor("linear", cats, nums, cfg)
    if family == "spline":
        # Periodic splines for the hour (it wraps around midnight), ordinary splines for other
        # continuous inputs, flags and one-hot categories unchanged: one variable can span many columns.
        binary = {"is_weekend", "rush_hour", "is_holiday", "is_raining", "is_snowing"}
        parts = []
        if "hour" in nums:
            parts.append(("hour", SplineTransformer(n_knots=13, degree=3, extrapolation="periodic"), ["hour"]))
        smooth = [n for n in nums if n not in binary and n not in ("hour", "hour_sin", "hour_cos")]
        if smooth:
            parts.append(("smooth", SplineTransformer(n_knots=8, degree=3), smooth))
        flags = [n for n in nums if n in binary]
        if flags:
            parts.append(("flags", "passthrough", flags))
        if cats:
            parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
        return Pipeline([("prep", ColumnTransformer(parts)), ("model", Ridge(alpha=1e-3))])
    raise ValueError(family)


def model_features(family: str, variables: list[str]) -> list[str]:
    """``linear_raw`` sees only each variable's untransformed column; other families see all its features."""
    return [RAW_FEATURE_OF[v] for v in variables] if family == "linear_raw" else features_using_only(variables)


def variable_models(X: pd.DataFrame, y: pd.Series, splits: dict, ranking: pd.DataFrame, params: dict,
                    cfg: Config, tree_kind: str = "lgbm") -> dict:
    """Univariate, bivariate and multivariate models, fitted on training years and scored on validation."""
    order = list(ranking.index)
    top1, top2 = order[:1], order[:2]
    tr, va = splits["train"], splits["validation"]
    rows, curves = [], {}
    sets = [("univariate", top1), ("bivariate", top2), ("multivariate", list(RAW_VARIABLES)),
            ("univariate (raw weather)", [cfg.feature_study.contrast_variable])]
    for label, variables in sets:
        families = ["linear_raw", "linear", "spline", "tree"] if label.startswith("univariate") else ["linear", "tree"]
        for family in families:
            model = family_model(family, variables, params, cfg, tree_kind)
            Xs = X[model_features(family, variables)]
            model.fit(Xs.iloc[tr], y.iloc[tr])
            pred = model.predict(Xs.iloc[va])
            rows.append({"model": label, "variables": ", ".join(variables), "family": family,
                         "features_used": len(Xs.columns), **regression_metrics(y.iloc[va], pred)})
            if label.startswith("univariate"):
                curves[f"{label}|{family}"] = {"variable": variables[0], "model": model, "columns": list(Xs.columns)}
    incremental = []
    for k in range(1, len(order) + 1):
        variables = order[:k]
        model = family_model("tree", variables, params, cfg, tree_kind)
        Xs = X[model_features("tree", variables)]
        model.fit(Xs.iloc[tr], y.iloc[tr])
        incremental.append({"variables": k, "added": order[k - 1],
                            **regression_metrics(y.iloc[va], model.predict(Xs.iloc[va]))})
    return {"top1": top1, "top2": top2, "table": pd.DataFrame(rows), "incremental": pd.DataFrame(incremental),
            "curves": curves}


def univariate_curves(curves: dict, X_val: pd.DataFrame, y_val: pd.Series, variable: str) -> dict:
    """What each univariate family predicts for every hour of the day, next to the validation year's
    mean and its 10th and 90th percentiles (the data behind the univariate figure)."""
    if variable != "hour":
        return {}
    h = np.arange(24)
    grid = pd.DataFrame({"hour": h.astype(float), "hour_sin": np.sin(2 * np.pi * h / 24),
                         "hour_cos": np.cos(2 * np.pi * h / 24)})
    by_hour = y_val.groupby(X_val["hour"].astype(int).to_numpy())
    out = {"hour": h.tolist(), "actual_mean": by_hour.mean().reindex(h).tolist(),
           "actual_p10": by_hour.quantile(0.1).reindex(h).tolist(), "actual_p90": by_hour.quantile(0.9).reindex(h).tolist()}
    for key, c in curves.items():
        label, family = key.split("|")
        if label == "univariate":
            out[family] = [float(v) for v in c["model"].predict(grid[c["columns"]])]
    return out


def group_ablation(X: pd.DataFrame, y: pd.Series, splits: dict, clean: pd.DataFrame, params: dict, cfg: Config,
                   tree_kind: str) -> pd.DataFrame:
    """Does each group of inputs earn its place? The champion's family and settings on growing sets of
    variables, scored on the validation year: overall, and on the hours with rain, storm or snow, where
    weather inputs ought to matter most."""
    tr, va = splits["train"], splits["validation"]
    kinds = weather_kinds(clean.iloc[va])
    wet = np.isin(kinds, ["Rain or storm", "Snow"])
    rows = []
    for label, variables in INPUT_GROUPS:
        cols = model_features("tree", variables)
        model = family_model("tree", variables, params, cfg, tree_kind).fit(X.iloc[tr][cols], y.iloc[tr])
        pred = model.predict(X.iloc[va][cols])
        actual = y.iloc[va].to_numpy()
        rows.append({"inputs": label, "variables": len(variables), "features_used": len(cols),
                     **regression_metrics(actual, pred), "rmse_rain_storm_snow": rmse_of(actual[wet], pred[wet]),
                     "rmse_dry": rmse_of(actual[~wet], pred[~wet])})
    out = pd.DataFrame(rows)
    out.attrs["wet_hours"] = int(wet.sum())
    return out


def bivariate_surface(X: pd.DataFrame, y: pd.Series, splits: dict, top2: list[str], params: dict,
                      cfg: Config, tree_kind: str = "lgbm") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mean actual and predicted volume by weekday and hour on the validation year (bivariate tree)."""
    cols = model_features("tree", top2)
    model = family_model("tree", top2, params, cfg, tree_kind).fit(X.iloc[splits["train"]][cols], y.iloc[splits["train"]])
    Xv = X.iloc[splits["validation"]]
    frame = pd.DataFrame({"weekday": Xv["weekday"].to_numpy(), "hour": Xv["hour"].to_numpy(),
                          "actual": y.iloc[splits["validation"]].to_numpy(), "predicted": model.predict(Xv[cols])})
    table = frame.groupby(["weekday", "hour"])[["actual", "predicted"]].mean()
    return table["actual"].unstack("hour"), table["predicted"].unstack("hour")
