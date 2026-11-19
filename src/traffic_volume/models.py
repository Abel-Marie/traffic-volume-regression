"""Candidate regressors, each wrapped in the preprocessing it needs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .config import Config
from .features import make_linear_preprocessor, make_tree_preprocessor


class HourOfWeekBaseline(BaseEstimator, RegressorMixin):
    """Predicts the training mean volume for the same hour of the same weekday.

    A strong naive forecast for traffic: it already captures the daily and
    weekly rhythm, so a model must beat it to be worth deploying.
    """

    def fit(self, X: pd.DataFrame, y):
        frame = pd.DataFrame({"weekday": X["weekday"].to_numpy(), "hour": X["hour"].to_numpy(),
                              "y": np.asarray(y, dtype=float)})
        self.table_ = frame.groupby(["weekday", "hour"])["y"].mean().to_dict()
        self.fallback_ = float(frame["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        keys = zip(X["weekday"].to_numpy(), X["hour"].to_numpy())
        return np.array([self.table_.get(k, self.fallback_) for k in keys])


class HolidayAwareBaseline(BaseEstimator, RegressorMixin):
    """The same lookup table with one more rule: a quiet holiday is its own, eighth day type.

    Still a table of averages (8 day types x 24 hours). It shows how much of a model's advantage over
    the plain baseline is simply knowing which days are holidays.
    """

    def fit(self, X: pd.DataFrame, y):
        frame = pd.DataFrame({"day": np.where(X["is_holiday"].to_numpy() == 1, 7, X["weekday"].to_numpy()),
                              "weekday": X["weekday"].to_numpy(), "hour": X["hour"].to_numpy(),
                              "y": np.asarray(y, dtype=float)})
        self.table_ = frame.groupby(["day", "hour"])["y"].mean().to_dict()
        self.by_weekday_ = frame.groupby(["weekday", "hour"])["y"].mean().to_dict()
        self.fallback_ = float(frame["y"].mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        day = np.where(X["is_holiday"].to_numpy() == 1, 7, X["weekday"].to_numpy())
        keys = zip(day, X["weekday"].to_numpy(), X["hour"].to_numpy())
        return np.array([self.table_.get((d, h), self.by_weekday_.get((w, h), self.fallback_)) for d, w, h in keys])


LINEAR_KINDS = {"linear", "ridge", "lasso", "poly_ridge", "mlp"}

MODEL_NOTES = {
    "baseline": "Mean of the same hour and weekday in training. The bar to beat.",
    "baseline_holiday": "The same lookup table with quiet holidays as an eighth day type.",
    "linear": "Ordinary least squares; each coefficient is directly readable.",
    "ridge": "Linear with L2 shrinkage; stable when features are correlated.",
    "lasso": "Linear with L1 shrinkage; drops weak features.",
    "poly_ridge": "Ridge on squares and pairwise products of the numbers: curved effects and interactions.",
    "tree": "One readable regression tree; unstable and prone to overfitting.",
    "rf": "Averages many trees; robust but large and slower to score.",
    "hgb": "Gradient boosting in scikit-learn; strong and fast.",
    "lgbm": "Gradient boosting in LightGBM; usually strongest on tabular data. Explained with SHAP.",
    "mlp": "Small neural network on scaled inputs and a scaled target; hardest to interpret.",
}


def base_regressor(kind: str, cfg: Config, **params):
    seed, jobs = cfg.seed, cfg.models.n_jobs
    if kind == "linear":
        return LinearRegression()
    if kind in ("ridge", "poly_ridge"):
        return Ridge(**{"alpha": 1.0, **params})
    if kind == "lasso":
        return Lasso(**{"alpha": 1.0, "max_iter": 20000, **params})
    if kind == "tree":
        return DecisionTreeRegressor(random_state=seed, **{"min_samples_leaf": 20, **params})
    if kind == "rf":
        return RandomForestRegressor(random_state=seed, n_jobs=jobs,
                                     **{"n_estimators": 200, "min_samples_leaf": 5, "max_features": 0.5, **params})
    if kind == "hgb":
        # No early stopping: it would hold out a random tenth of the training hours, which sit next to
        # hours the model trains on. The number of boosting rounds is tuned by time-ordered CV instead.
        return HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **params)
    if kind == "lgbm":
        return LGBMRegressor(objective="regression", random_state=seed, n_jobs=jobs, deterministic=True,
                             force_row_wise=True, verbose=-1,
                             **{"n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, **params})
    if kind == "mlp":
        net = MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=300,
                           random_state=seed, **params)
        return TransformedTargetRegressor(regressor=net, transformer=StandardScaler())
    raise ValueError(f"unknown model kind: {kind}")


def make_regressor(kind: str, cats: list[str], nums: list[str], cfg: Config, **params) -> Pipeline:
    """Preprocessing plus estimator. Linear models get one-hot and scaling; trees get integer codes."""
    if kind == "baseline":
        return Pipeline([("model", HourOfWeekBaseline())])
    if kind == "baseline_holiday":
        return Pipeline([("model", HolidayAwareBaseline())])
    minf = cfg.features.min_category_frequency
    if kind in LINEAR_KINDS:
        prep = make_linear_preprocessor(cats, nums, minf, degree=2 if kind == "poly_ridge" else 1)
    else:
        prep = make_tree_preprocessor(cats, nums, minf)
    return Pipeline(prep + [("model", base_regressor(kind, cfg, **params))])


TREE_KINDS = {"lgbm", "rf", "hgb", "tree"}
MODEL_LABELS = {"baseline": "Hour-of-week average", "baseline_holiday": "Hour-of-week average + holidays",
                "linear": "Linear regression", "ridge": "Ridge", "lasso": "Lasso", "poly_ridge": "Polynomial ridge",
                "tree": "Decision tree", "rf": "Random forest", "hgb": "HistGradientBoosting", "lgbm": "LightGBM",
                "mlp": "Neural network"}
