"""The deployed service must reproduce the pipeline's predictions (within 0.01 vehicles per hour).

Goes through the API's own input schema: data row -> JSON payload -> HourRequest -> raw variables ->
shared features -> ONNX. Both exported models are checked: the champion (requests that carry the
weather) and the calendar-only model (requests that do not).
"""

import numpy as np
import pandas as pd
import pytest

from traffic_volume.monitoring.latency import to_api_payload


@pytest.fixture(scope="module")
def parity(root, service_app):
    export = root / "artifacts" / "export"
    if not (export / "parity_rows.pkl").exists():
        pytest.skip("run the pipeline first")
    from core.schemas import HourRequest

    rows = pd.read_pickle(export / "parity_rows.pkl")

    def served(weather: bool) -> np.ndarray:
        requests = [HourRequest(**to_api_payload(r, weather=weather)) for r in rows.to_dict("records")]
        items = [{"moment": q.datetime, "is_holiday": q.is_holiday, "weather": q.weather.raw() if q.weather else None}
                 for q in requests]
        return np.array([p["raw_prediction"] for p in service_app.MODEL.predict(items)])

    return export, rows, served


@pytest.mark.needs_artifacts
def test_champion_matches_pipeline(parity):
    export, rows, served = parity
    expected = np.load(export / "parity_expected.npy")
    assert len(rows) >= 200
    assert np.max(np.abs(served(weather=True) - expected)) <= 0.01


@pytest.mark.needs_artifacts
def test_calendar_only_model_matches_pipeline(parity, service_app):
    export, rows, served = parity
    expected = np.load(export / "parity_expected_calendar.npy")
    if not len(expected):
        pytest.skip("the champion uses the calendar only: there is no second model")
    assert service_app.MODEL.calendar is not None
    assert np.max(np.abs(served(weather=False) - expected)) <= 0.01


@pytest.mark.needs_artifacts
def test_the_service_names_the_same_holidays_as_the_cleaned_data(parity, service_app):
    """The API works the holiday out from the date; training took it from the cleaned data. Same answer?"""
    _, rows, _ = parity
    for stamp, quiet in zip(rows["date_time"], rows["quiet_holiday"]):
        assert service_app.MODEL.holiday(stamp, None)["counts_as_holiday"] == bool(quiet)
