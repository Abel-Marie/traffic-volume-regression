"""Typed configuration objects, loaded from ``configs/config.yaml``.

Loading fails loudly when a key is missing or unexpected, so a typo in the
YAML file cannot silently fall back to a default.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    name: str
    citation: str
    license: str
    sources: list[str]
    mirror_path: str
    download_deadline_seconds: int
    sha256: str
    expected_rows: int
    expected_columns: int


@dataclass(frozen=True)
class CleaningConfig:
    min_valid_temp_kelvin: float
    max_valid_rain_mm: float
    min_plausible_volume: int
    quiet_holidays: list[str]
    weather_severity: list[str]


@dataclass(frozen=True)
class SplitConfig:
    train_until_year: int
    validation_year: int
    test_year: int


@dataclass(frozen=True)
class FeatureConfig:
    min_category_frequency: int
    sources: list[str]


@dataclass(frozen=True)
class ModelsConfig:
    n_jobs: int


@dataclass(frozen=True)
class TuningConfig:
    cv_folds: int
    ridge_grid: dict[str, list[Any]]
    rf_random_iterations: int
    hgb_random_iterations: int
    lgbm_budget: int
    lgbm_grid: dict[str, list[Any]]


@dataclass(frozen=True)
class DiagnosticsConfig:
    learning_curve_sizes: list[float]
    complexity_curve: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class EvaluationConfig:
    bootstrap_samples: int


@dataclass(frozen=True)
class ExplainConfig:
    permutation_repeats: int
    shap_rows: int


@dataclass(frozen=True)
class FeatureStudyConfig:
    univariate_families: list[str]
    contrast_variable: str


@dataclass(frozen=True)
class SuccessConfig:
    r2: float
    beat_baseline_rmse: bool


@dataclass(frozen=True)
class SelectionConfig:
    tie_tolerance_rmse: float


@dataclass(frozen=True)
class ServiceConfig:
    url: str


@dataclass(frozen=True)
class Config:
    project: str
    seed: int
    data: DataConfig
    target: str
    cleaning: CleaningConfig
    split: SplitConfig
    features: FeatureConfig
    models: ModelsConfig
    tuning: TuningConfig
    diagnostics: DiagnosticsConfig
    evaluation: EvaluationConfig
    explain: ExplainConfig
    feature_study: FeatureStudyConfig
    success: SuccessConfig
    selection: SelectionConfig
    test_pass_models: list[str]
    first_evaluation: dict[str, Any]
    service: ServiceConfig
    fast: dict[str, Any]
    is_fast: bool = False


SECTION_TYPES = {
    "data": DataConfig, "cleaning": CleaningConfig, "split": SplitConfig, "features": FeatureConfig,
    "models": ModelsConfig, "tuning": TuningConfig, "diagnostics": DiagnosticsConfig,
    "evaluation": EvaluationConfig, "explain": ExplainConfig, "feature_study": FeatureStudyConfig,
    "success": SuccessConfig, "selection": SelectionConfig, "service": ServiceConfig,
}


def build_section(cls: type, values: dict[str, Any], section: str) -> Any:
    """Instantiate one config dataclass, rejecting missing or unknown keys."""
    fields = dataclasses.fields(cls)
    allowed = {f.name for f in fields}
    required = {f.name for f in fields
                if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING}
    missing, unknown = required - set(values), set(values) - allowed
    if missing or unknown:
        raise ValueError(f"config section '{section}': missing={sorted(missing)} unknown={sorted(unknown)}")
    return cls(**values)


def config_from_dict(raw: dict[str, Any]) -> Config:
    """Convert the parsed YAML mapping into a validated :class:`Config`."""
    raw = dict(raw)
    sections = {name: build_section(cls, raw.pop(name), name) for name, cls in SECTION_TYPES.items()}
    cfg = build_section(Config, {**raw, **sections}, "top level")
    grid = cfg.tuning.lgbm_grid
    if len(grid["num_leaves"]) * len(grid["learning_rate"]) * len(grid["n_estimators"]) != cfg.tuning.lgbm_budget:
        raise ValueError("tuning.lgbm_grid must contain exactly lgbm_budget combinations")
    return cfg


def load_config_text(text: str) -> Config:
    """Parse configuration from a YAML string (used by the self-contained notebook)."""
    return config_from_dict(yaml.safe_load(text))


def load_config(path: str | Path) -> Config:
    return load_config_text(Path(path).read_text(encoding="utf-8"))


def apply_fast_mode(cfg: Config) -> Config:
    """Return a copy with tiny budgets for smoke tests. Never used for reported results."""
    f = cfg.fast
    return dataclasses.replace(
        cfg, is_fast=True,
        tuning=dataclasses.replace(cfg.tuning, cv_folds=f["cv_folds"], rf_random_iterations=f["rf_random_iterations"],
                                   hgb_random_iterations=f["hgb_random_iterations"], lgbm_budget=f["lgbm_budget"],
                                   lgbm_grid={"num_leaves": [15, 31], "learning_rate": [0.1], "n_estimators": [50, 100]}),
        diagnostics=dataclasses.replace(
            cfg.diagnostics, learning_curve_sizes=[0.3, 1.0],
            complexity_curve={k: {"param": v["param"], "values": v["values"][1::4]}
                              for k, v in cfg.diagnostics.complexity_curve.items()}),
        evaluation=dataclasses.replace(cfg.evaluation, bootstrap_samples=f["bootstrap_samples"]),
        explain=ExplainConfig(permutation_repeats=f["permutation_repeats"], shap_rows=f["shap_rows"]),
    )
