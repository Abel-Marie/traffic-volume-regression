"""End to end: every stage in fast mode, in a scratch copy of the project."""

import json
import shutil

import pytest

from traffic_volume.cli import main


@pytest.mark.slow
def test_all_stages_fast(root, tmp_path):
    shutil.copytree(root / "configs", tmp_path / "configs")
    shutil.copytree(root / "data" / "mirror", tmp_path / "data" / "mirror")
    main(["all", "--fast", "--root", str(tmp_path)])
    out = tmp_path / "build" / "fast"  # a fast run never writes where the reported results live
    assert not (tmp_path / "reports").exists()
    report = json.loads((out / "reports" / "metrics" / "pipeline" / "metrics.json").read_text())
    assert report["fast_mode"] is True
    assert list(report["test"]["models"]) == ["baseline", "baseline_holiday", "ridge", "univariate", "bivariate",
                                              "alternative_inputs", "champion"]
    assert report["champion"]["label"] in {r["model"] for r in report["champion"]["comparison"]}
    assert report["deployment"]["parity_ok"] is True
    assert set(report["test"]["slices"]) == {"by_segment", "by_month", "by_weather", "by_period"}
    assert len(list((out / "reports" / "figures").glob("*.png"))) >= 28
    first = (out / "artifacts" / "export" / "model.onnx").read_bytes()
    main(["export", "--fast", "--root", str(tmp_path)])
    assert (out / "artifacts" / "export" / "model.onnx").read_bytes() == first, "the export must be reproducible"
