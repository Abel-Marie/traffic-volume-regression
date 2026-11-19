"""Where every stage reads and writes, plus JSON helpers and the run manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class RunContext:
    """Folder layout for one run.

    ``layout="pipeline"`` follows the repository layout (data/, artifacts/,
    reports/). ``layout="notebook"`` keeps everything under one output folder
    so a notebook works when it is the only file present.
    """

    root: Path
    layout: str = "pipeline"
    show: bool = False
    repo: Path | None = None  # where configs/ and data/mirror/ live, when the outputs go elsewhere

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.repo = Path(self.repo) if self.repo is not None else self.root
        if self.layout not in {"pipeline", "notebook"}:
            raise ValueError("layout must be 'pipeline' or 'notebook'")

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def fig_dir(self) -> Path:
        return self.root / "reports" / "figures" if self.layout == "pipeline" else self.root / "figures"

    @property
    def metrics_dir(self) -> Path:
        return self.root / "reports" / "metrics" / "pipeline" if self.layout == "pipeline" else self.root

    def ensure(self) -> RunContext:
        for d in (self.data_dir / "raw", self.data_dir / "interim", self.data_dir / "processed",
                  self.artifacts_dir, self.fig_dir, self.metrics_dir):
            d.mkdir(parents=True, exist_ok=True)
        return self


def to_jsonable(obj: Any) -> Any:
    """Convert numpy scalars/arrays and paths so ``json.dump`` accepts them."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and obj != obj:
        return None
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(obj), indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions(names: list[str]) -> dict[str, str]:
    out = {}
    for name in names:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


TRACKED_PACKAGES = ["numpy", "pandas", "scikit-learn", "lightgbm", "optuna", "shap", "statsmodels",
                    "matplotlib", "seaborn", "skl2onnx", "onnxmltools", "onnxruntime"]


def git_sha(root: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(ctx: RunContext, stage: str, config_dict: dict[str, Any], extra: dict[str, Any]) -> Path:
    """Record what produced an artifact: code version, config, environment, results."""
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    manifest = {
        "stage": stage,
        "created_utc": stamp,
        "git_sha": git_sha(ctx.repo),
        "python": platform.python_version(),
        "packages": package_versions(TRACKED_PACKAGES),
        "config": config_dict,
        **extra,
    }
    return save_json(manifest, ctx.artifacts_dir / "runs" / f"{stamp}_{stage}.json")
