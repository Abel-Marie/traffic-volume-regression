"""Shared fixtures. Tests that need a finished pipeline run skip cleanly without one."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "service"
sys.path.insert(0, str(ROOT / "src"))

from traffic_volume.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg():
    return load_config(ROOT / "configs" / "config.yaml")


@pytest.fixture(scope="session")
def sample_raw() -> pd.DataFrame:
    """First 3,000 rows of the shipped data mirror, with row ids."""
    df = pd.read_csv(ROOT / "data" / "mirror" / "traffic.csv.gz", nrows=3000, keep_default_na=False, na_values=[""])
    df.insert(0, "row_id", range(len(df)))
    return df


def service_ready() -> bool:
    return (SERVICE / "models" / "model.onnx").exists() and (SERVICE / "core" / "features.py").exists()


@pytest.fixture(scope="session")
def service_app():
    if not service_ready():
        pytest.skip("service not synced: run `traffic-volume all`, then `python scripts/sync_service.py`")
    sys.path.insert(0, str(SERVICE))
    import main  # noqa: PLC0415

    return main
