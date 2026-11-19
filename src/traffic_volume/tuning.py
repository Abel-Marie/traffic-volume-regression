"""Hyperparameter search: Grid, Random and Bayesian (Optuna TPE) behind one interface.

All searches use expanding-window time-series cross-validation on the training
years, fixed trial counts and seeded samplers, so repeated runs agree. Scores
are RMSE in vehicles per hour (lower is better).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import optuna
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score

from .config import Config
from .models import make_regressor
from .split import time_series_cv


def search_summary(name: str, model: str, rmses: list[float], params: list[dict], seconds: float,
                   folds: int) -> dict:
    best = int(np.argmin(rmses))
    return {"strategy": name, "model": model, "configs_tried": len(rmses), "fits": len(rmses) * folds,
            "seconds": round(seconds, 1), "best_rmse": float(rmses[best]), "best_params": params[best],
            "history": [float(v) for v in rmses], "best_so_far": [float(v) for v in np.minimum.accumulate(rmses)]}


def unprefix(params: dict) -> dict:
    return {k.split("__", 1)[1]: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def grid_search_rmse(kind: str, grid: dict, X, y, cats, nums, cfg: Config) -> dict:
    start = time.time()
    search = GridSearchCV(make_regressor(kind, cats, nums, cfg), {f"model__{k}": v for k, v in grid.items()},
                          scoring="neg_root_mean_squared_error", cv=time_series_cv(cfg), n_jobs=1, refit=False)
    search.fit(X, y)
    r = search.cv_results_
    return search_summary("grid", kind, list(-r["mean_test_score"]), [unprefix(p) for p in r["params"]],
                          time.time() - start, cfg.tuning.cv_folds)


def random_search_rmse(kind: str, space: dict, n_iter: int, X, y, cats, nums, cfg: Config) -> dict:
    start = time.time()
    search = RandomizedSearchCV(make_regressor(kind, cats, nums, cfg), {f"model__{k}": v for k, v in space.items()},
                                n_iter=n_iter, scoring="neg_root_mean_squared_error", cv=time_series_cv(cfg),
                                n_jobs=1, refit=False, random_state=cfg.seed)
    search.fit(X, y)
    r = search.cv_results_
    return search_summary("random", kind, list(-r["mean_test_score"]), [unprefix(p) for p in r["params"]],
                          time.time() - start, cfg.tuning.cv_folds)


def lgbm_space() -> dict:
    """Random-search space for LightGBM; the Optuna space covers the same ranges."""
    return {"num_leaves": randint(15, 256), "learning_rate": loguniform(0.01, 0.2), "n_estimators": randint(100, 1200),
            "min_child_samples": randint(5, 100), "subsample": uniform(0.5, 0.5), "subsample_freq": [1],
            "colsample_bytree": uniform(0.5, 0.5), "reg_lambda": loguniform(1e-3, 10.0)}


def lgbm_trial(trial: optuna.Trial) -> dict:
    return {"num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1199),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 99),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)}


def bayesian_search_rmse(n_trials: int, X, y, cats, nums, cfg: Config) -> dict:
    """Optuna TPE with a seeded sampler and one worker (reproducible)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    start = time.time()
    cv = time_series_cv(cfg)

    def objective(trial: optuna.Trial) -> float:
        model = make_regressor("lgbm", cats, nums, cfg, **lgbm_trial(trial))
        return float(-np.mean(cross_val_score(model, X, y, scoring="neg_root_mean_squared_error", cv=cv, n_jobs=1)))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=cfg.seed))
    study.optimize(objective, n_trials=n_trials, n_jobs=1)
    return search_summary("bayesian", "lgbm", [t.value for t in study.trials],
                          [{**t.params, "subsample_freq": 1} for t in study.trials], time.time() - start,
                          cfg.tuning.cv_folds)


def rf_space() -> dict:
    """Kept moderate (fewer, shallower trees) so the search fits a single-core budget; the forest is a
    comparison here, and deep forests are too large to deploy anyway."""
    return {"n_estimators": randint(100, 201), "max_depth": [12, 16, 20], "min_samples_leaf": randint(3, 11),
            "max_features": [0.3, 0.5]}


def hgb_space() -> dict:
    """Random-search space for HistGradientBoosting (rounds, learning rate, tree size, regularisation)."""
    return {"max_iter": randint(50, 401), "learning_rate": loguniform(0.02, 0.3), "max_leaf_nodes": randint(8, 65),
            "min_samples_leaf": randint(10, 101), "l2_regularization": loguniform(1e-3, 10.0)}


def run_all_searches(X, y, cats, nums, cfg: Config, cache_dir: Path | None = None) -> dict:
    """Grid search on ridge, random search on the random forest and on HistGradientBoosting, and all
    three strategies on LightGBM with the same budget.

    With ``cache_dir``, each finished search is saved immediately and skipped on a re-run, so an
    interrupted tuning stage resumes where it stopped. The file names carry a fingerprint of the
    training data and the search settings, so a result from different data or budgets is never reused.
    """
    t = cfg.tuning
    fingerprint = hashlib.sha256(repr((len(X), round(float(np.sum(y)), 3), list(X.columns), t, cfg.seed,
                                       cfg.features.min_category_frequency)).encode()).hexdigest()[:10]
    jobs = {
        "ridge_grid": lambda: grid_search_rmse("ridge", t.ridge_grid, X, y, cats, nums, cfg),
        "rf_random": lambda: random_search_rmse("rf", rf_space(), t.rf_random_iterations, X, y, cats, nums, cfg),
        "hgb_random": lambda: random_search_rmse("hgb", hgb_space(), t.hgb_random_iterations, X, y, cats, nums, cfg),
        "lgbm_grid": lambda: grid_search_rmse("lgbm", t.lgbm_grid, X, y, cats, nums, cfg),
        "lgbm_random": lambda: random_search_rmse("lgbm", lgbm_space(), t.lgbm_budget, X, y, cats, nums, cfg),
        "lgbm_bayesian": lambda: bayesian_search_rmse(t.lgbm_budget, X, y, cats, nums, cfg),
    }
    results = {}
    for name, run in jobs.items():
        cached = Path(cache_dir) / f"{name}-{fingerprint}.json" if cache_dir else None
        if cached is not None and cached.exists():
            results[name] = json.loads(cached.read_text())
            continue
        results[name] = run()
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(results[name], default=float))
    return results


def best_lgbm(tuning: dict) -> dict:
    return min((tuning[k] for k in ("lgbm_grid", "lgbm_random", "lgbm_bayesian")), key=lambda r: r["best_rmse"])
