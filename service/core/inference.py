"""Load the exported models once and turn conditions into predictions.

Steps, identical to the training pipeline: shared feature code -> category
codes from metadata.json -> ONNX model.

Two models may be loaded: the champion, and the calendar-only model that answers when a request
carries no weather. Which one answered is reported with every prediction.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .features import derive_features, holiday_name

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


class ExportedModel:
    """One ONNX file and the description that says how to feed it."""

    def __init__(self, block: dict, model_dir: Path) -> None:
        options = ort.SessionOptions()
        options.log_severity_level = 3
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_dir / block["file"]), options, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.order, self.codes, self.unknown = block["input_order"], block["category_codes"], block["unknown_category_code"]
        self.sources, self.version, self.inputs = block["feature_sources"], block["model_version"], block["inputs"]
        self.typical_error = int(round(block["test_metrics"]["rmse"]))

    def raw_predictions(self, records: list[dict]) -> np.ndarray:
        rows = []
        for r in records:
            f = derive_features(r, self.sources)
            rows.append([self.codes[c].get(str(f[c]), self.unknown) if c in self.codes else float(f[c])
                         for c in self.order])
        X = np.asarray(rows, dtype=np.float32)
        return np.asarray(self.session.run(None, {self.input_name: X})[0], dtype=float).ravel()


class TrafficModel:
    def __init__(self, model_dir: Path = MODEL_DIR) -> None:
        self.metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        self.primary = ExportedModel(self.metadata, model_dir)
        fallback = self.metadata.get("fallback")
        self.calendar = ExportedModel(fallback, model_dir) if fallback else None
        self.quiet_holidays = set(self.metadata["quiet_holidays"])
        self.needs_weather = self.primary.inputs != "calendar only"

    @property
    def version(self) -> str:
        return self.primary.version

    def holiday(self, moment, override: bool | None) -> dict:
        """The named day on that date and whether it counts as a quiet holiday. The calendar decides
        unless the request says otherwise."""
        name = holiday_name(moment.year, moment.month, moment.day)
        name = None if name == "None" else name
        if override is None:
            return {"name": name, "counts_as_holiday": name in self.quiet_holidays, "decided_by": "calendar"}
        return {"name": name, "counts_as_holiday": bool(override), "decided_by": "request"}

    def model_for(self, has_weather: bool) -> ExportedModel:
        """The champion when the weather is given (or when it never needs it); otherwise the calendar-only model."""
        if has_weather or not self.needs_weather or self.calendar is None:
            return self.primary
        return self.calendar

    def predict(self, items: list[dict]) -> list[dict]:
        """``items``: dicts with ``moment`` (a datetime), ``is_holiday`` (bool or None) and ``weather``
        (raw weather variables, or None). Returns one prediction per item, in order."""
        out: list[dict | None] = [None] * len(items)
        groups: dict[int, tuple[ExportedModel, list[int], list[dict]]] = {}
        for i, item in enumerate(items):
            moment, weather = item["moment"], item.get("weather")
            holiday = self.holiday(moment, item.get("is_holiday"))
            model = self.model_for(weather is not None)
            record = {"hour": moment.hour, "weekday": moment.weekday(), "month": moment.month,
                      "is_holiday": int(holiday["counts_as_holiday"]), **(weather or {})}
            out[i] = {"prediction_id": uuid.uuid4().hex, "datetime": moment.replace(minute=0, second=0, microsecond=0),
                      "model_used": model.inputs, "model_version": model.version,
                      "typical_error": model.typical_error, "holiday": holiday}
            groups.setdefault(id(model), (model, [], []))
            groups[id(model)][1].append(i)
            groups[id(model)][2].append(record)
        for model, positions, records in groups.values():
            for i, value in zip(positions, model.raw_predictions(records)):
                out[i]["raw_prediction"] = round(float(value), 4)
                out[i]["vehicles_per_hour"] = int(round(max(float(value), 0.0)))
        return out
