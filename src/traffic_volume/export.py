"""Export the models for serving: ONNX files plus one metadata file.

The service reproduces the pipeline from these files: shared feature code
-> category codes (vocabulary below) -> ONNX model.

Two models are exported when the champion uses the weather: the champion itself, and the same model
trained on the calendar inputs alone. The service answers with the second one when a request carries
no weather, which is the normal case when planning further ahead than a weather forecast reaches.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from .config import Config
from .rowlevel import FEATURE_INPUTS, FEATURE_SOURCES, RAW_VARIABLES, derive_features

WEATHER_VARIABLES = ["temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"]


def category_codes(model, cats: list[str]) -> dict[str, dict[str, float]]:
    """The code the fitted encoder gives each training category (grouped rare ones share a code)."""
    if not cats:
        return {}
    encoder = model.named_steps["prep"].named_transformers_["cat"]
    vocab = {}
    for j, col in enumerate(cats):
        known = list(encoder.categories_[j])
        probe = pd.DataFrame({c: [encoder.categories_[i][0]] * len(known) for i, c in enumerate(cats)})
        probe[col] = known
        vocab[col] = {str(k): float(v) for k, v in zip(known, encoder.transform(probe)[:, j])}
    return vocab


@contextmanager
def plain_tree_attributes():
    """Work around a mismatch between skl2onnx and the current ONNX library.

    For HistGradientBoosting, skl2onnx describes the trees with lists that contain booleans and NumPy
    integers; ONNX now refuses those where it expects plain integers ("Expected an int, got a boolean").
    While a conversion runs, every such value is turned into a plain int before it reaches ONNX. The
    values themselves are unchanged, and the parity check after every export confirms the result.
    """
    from skl2onnx.common._container import ModelComponentContainer

    original = ModelComponentContainer.add_node

    def add_node(self, *args, **attrs):
        fixed = {k: [int(v) if isinstance(v, (bool, np.bool_, np.integer)) else v for v in value]
                 if isinstance(value, list) else value for k, value in attrs.items()}
        return original(self, *args, **fixed)

    ModelComponentContainer.add_node = add_node
    try:
        yield
    finally:
        ModelComponentContainer.add_node = original


def to_onnx(model, n_features: int, path: Path) -> Path:
    """Convert the fitted regressor (after preprocessing) to ONNX with float32 input.

    The converters stamp each file with a random graph name and list the operator sets in no fixed
    order; both are fixed here so that the same model always gives the same bytes, and therefore the
    same version string in the notebook, the pipeline and the service.
    """
    est = model.named_steps["model"]
    if type(est).__name__ == "LGBMRegressor":
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType as LgbmFloat
        onx = convert_lightgbm(est, initial_types=[("input", LgbmFloat([None, n_features]))], target_opset=15)
    else:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        with plain_tree_attributes():
            onx = convert_sklearn(est, initial_types=[("input", FloatTensorType([None, n_features]))], target_opset=15)
    onx.graph.name = "traffic_volume"
    onx.doc_string = ""
    onx.producer_name, onx.producer_version = "traffic-volume", "1"
    opsets = sorted((o.domain, o.version) for o in onx.opset_import)
    del onx.opset_import[:]
    for domain, version in opsets:
        entry = onx.opset_import.add()
        entry.domain, entry.version = domain, version
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(onx.SerializeToString(deterministic=True))
    return path


def onnx_predict(path: Path, X32: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    return np.asarray(session.run(None, {session.get_inputs()[0].name: X32.astype(np.float32)})[0]).ravel()


def model_block(model, name: str, cats: list[str], nums: list[str], sources: list[str], test_metrics: dict,
                path: Path) -> dict:
    """Convert one fitted pipeline and describe it: everything the service needs to run that model.

    ``cats``, ``nums`` and ``sources`` describe the whole feature frame; the model's own inputs are read
    from the fitted pipeline, which may use fewer."""
    used = list(model.named_steps["prep"].get_feature_names_out())
    to_onnx(model, len(used), path)
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]
    cats, nums = [c for c in cats if c in used], [c for c in nums if c in used]
    needed = {v for feature in used for v in FEATURE_INPUTS[feature]}
    variables = [v for v in RAW_VARIABLES if v in needed]
    return {
        "model_version": f"{name}-{digest}", "file": Path(path).name,
        "inputs": "calendar and weather" if set(variables) & set(WEATHER_VARIABLES) else "calendar only",
        "input_variables": variables,
        "feature_sources": [s for s in sources if set(FEATURE_SOURCES[s](dict.fromkeys(RAW_VARIABLES, 0))) & set(used)],
        "input_order": used, "categorical_features": cats, "numeric_features": nums, "unknown_category_code": -1.0,
        "category_codes": category_codes(model, cats), "test_metrics": test_metrics,
    }


def export_metadata(primary: dict, fallback: dict | None, champion: dict, cfg: Config,
                    reference: dict | None = None) -> dict:
    """The metadata file. The champion's block sits at the top level; ``fallback`` is the calendar-only
    model, present when the champion needs the weather; ``reference`` holds the test results of the two
    lookup tables, so that the service can show the model's error next to theirs."""
    return {
        **primary,
        "validation_metrics": {k: champion[f"val_{k}"] for k in ("mse", "rmse", "mae", "r2")},
        "test_reference": reference or {},
        "test_year_evaluations": 2,
        "first_evaluation": cfg.first_evaluation,
        "trained_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "task": "regression: westbound I-94 vehicles per hour at ATR station 301 (Minneapolis-St Paul)",
        "estimator": champion["name"], "params": champion["params"],
        "trained_on": f"hours up to {cfg.split.validation_year} (training and validation years)",
        "quiet_holidays": list(cfg.cleaning.quiet_holidays),
        "inputs_note": "is_holiday means a quiet holiday (see quiet_holidays). Weather values are those of the hour "
                       "being predicted: observed in the training data, a forecast when planning ahead. Without "
                       "weather, the calendar-only model answers.",
        "not_used": ["year (trees cannot extrapolate to unseen years)",
                     "recent traffic counts: not known when planning days or weeks ahead"],
        "data": {"name": cfg.data.name, "citation": cfg.data.citation, "license": cfg.data.license},
        "fallback": fallback,
    }


def served_predictions(raw_vars: pd.DataFrame, block: dict, onnx_path: Path) -> np.ndarray:
    """Reproduce the service's computation from the exported files (pipeline-side check). ``block`` is
    the metadata, or its ``fallback`` part for the calendar-only model."""
    codes, unknown = block["category_codes"], block["unknown_category_code"]
    rows = []
    for r in raw_vars.to_dict("records"):
        f = derive_features(r, block["feature_sources"])
        rows.append([codes[c].get(str(f[c]), unknown) if c in codes else float(f[c]) for c in block["input_order"]])
    return onnx_predict(onnx_path, np.asarray(rows, dtype=np.float32))


def export_for_service(fitted: dict, champion: dict, test_models: dict, f: dict, cfg: Config, out_dir: Path,
                       parity_rows: int = 200) -> tuple[dict, dict]:
    """Write the ONNX files and the metadata, and check each exported model against its pipeline.

    ``fitted`` holds the test pass's fitted pipelines. Returns the metadata and a deployment summary;
    ``parity_ok`` is False if any exported model differs from its pipeline by more than the tolerance.
    """
    out_dir = Path(out_dir)
    X, raw = f["X_validation"].head(parity_rows), f["raw_validation"].head(parity_rows)

    def one(key: str, file: str) -> tuple[dict, dict]:
        block = model_block(fitted[key], champion["name"], f["cats"], f["nums"], f["sources"],
                            test_models[key]["metrics"], out_dir / file)
        diff = float(np.max(np.abs(fitted[key].predict(X) - served_predictions(raw, block, out_dir / file))))
        return block, {"model_kb": round((out_dir / file).stat().st_size / 1024, 1),
                       "model_version": block["model_version"], "inputs": block["inputs"],
                       "parity_max_abs_diff": diff}

    primary, summary = one("champion", "model.onnx")
    fallback, fallback_summary = (one("alternative_inputs", "model_calendar.onnx")
                                  if primary["inputs"] != "calendar only" else (None, None))
    worst = max(summary["parity_max_abs_diff"], fallback_summary["parity_max_abs_diff"] if fallback_summary else 0.0)
    deployment = {"format": "onnx", **summary, "parity_rows": len(X), "parity_tolerance": 0.01,
                  "parity_ok": worst <= 0.01, "fallback": fallback_summary}
    reference = {"hour_of_week_average": test_models["baseline"]["metrics"],
                 "hour_of_week_average_with_holidays": test_models["baseline_holiday"]["metrics"]}
    return export_metadata(primary, fallback, champion, cfg, reference), deployment
