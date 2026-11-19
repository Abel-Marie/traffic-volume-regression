"""The service's feature code and model files must be exact copies of the pipeline's."""

import hashlib
import json

import pytest


def test_feature_code_is_identical(root):
    service_file = root / "service" / "core" / "features.py"
    if not service_file.exists():
        pytest.skip("run scripts/sync_service.py")
    body = "".join(service_file.read_text(encoding="utf-8").splitlines(keepends=True)[2:])
    assert body == (root / "src" / "traffic_volume" / "rowlevel.py").read_text(encoding="utf-8"), \
        "service/core/features.py is stale: run scripts/sync_service.py"


@pytest.mark.needs_artifacts
def test_model_files_are_identical(root):
    exported, served = root / "artifacts" / "export", root / "service" / "models"
    if not (exported / "metadata.json").exists() or not (served / "metadata.json").exists():
        pytest.skip("run the pipeline and scripts/sync_service.py")
    metadata = json.loads((exported / "metadata.json").read_text(encoding="utf-8"))
    names = ["metadata.json", metadata["file"]] + ([metadata["fallback"]["file"]] if metadata["fallback"] else [])
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    for name in names:
        assert digest(exported / name) == digest(served / name), f"service/models/{name} is stale: run scripts/sync_service.py"
    assert sorted(p.name for p in served.glob("*.onnx")) == sorted(n for n in names if n.endswith(".onnx"))


@pytest.mark.needs_artifacts
def test_the_version_is_the_fingerprint_of_the_file(root):
    served = root / "service" / "models"
    if not (served / "metadata.json").exists():
        pytest.skip("run scripts/sync_service.py")
    metadata = json.loads((served / "metadata.json").read_text(encoding="utf-8"))
    for block in [metadata] + ([metadata["fallback"]] if metadata["fallback"] else []):
        digest = hashlib.sha256((served / block["file"]).read_bytes()).hexdigest()[:12]
        assert block["model_version"].endswith(digest)
