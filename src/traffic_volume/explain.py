"""Model explanations on validation data (never on training data)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from .config import Config


def permutation_frame(model, X_val: pd.DataFrame, y_val, cfg: Config) -> pd.DataFrame:
    """Increase in validation RMSE when each derived feature is shuffled."""
    r = permutation_importance(model, X_val, y_val, scoring="neg_root_mean_squared_error",
                               n_repeats=cfg.explain.permutation_repeats, random_state=cfg.seed, n_jobs=1)
    return (pd.DataFrame({"feature": X_val.columns, "rmse_increase": r.importances_mean, "std": r.importances_std})
            .sort_values("rmse_increase", ascending=False).reset_index(drop=True))


def shap_values_for(model, X_val: pd.DataFrame, cfg: Config) -> shap.Explanation:
    """SHAP values (vehicles per hour) for a sample of validation hours, for tree models."""
    sample = X_val.sample(n=min(cfg.explain.shap_rows, len(X_val)), random_state=cfg.seed)
    names = list(model.named_steps["prep"].get_feature_names_out())
    transformed = pd.DataFrame(model[:-1].transform(sample), columns=names, index=sample.index)
    return shap.TreeExplainer(model.named_steps["model"])(transformed)


def shap_frame(values: shap.Explanation) -> pd.DataFrame:
    return (pd.DataFrame({"feature": values.feature_names, "mean_abs_shap": np.abs(values.values).mean(0)})
            .sort_values("mean_abs_shap", ascending=False).reset_index(drop=True))
