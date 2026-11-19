"""The FastAPI service, exercised in-process."""

import pytest
from fastapi.testclient import TestClient

WEATHER = {"temp_c": 4.0, "weather_main": "Clouds", "clouds_all": 75}
ORDINARY = "2026-11-19T08:00"      # a Thursday
THANKSGIVING = "2026-11-26T08:00"  # the Thursday after


@pytest.fixture(scope="module")
def client(service_app):
    return TestClient(service_app.app, raise_server_exceptions=False)


def predict(client, **body):
    r = client.post("/api/v1/predict", json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_health_and_model_card(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model_version"]
    card = client.get("/api/v1/model").json()
    assert card["test_metrics"]["r2"] > 0.9
    assert card["test_reference"]["hour_of_week_average"]["rmse"] > card["test_metrics"]["rmse"]
    assert len(card["quiet_holidays"]) == 6 and card["first_evaluation"]["champion"] == "rf"
    for model in card["models"].values():
        assert model["version"] and model["test_metrics"]["rmse"] > 0


def test_a_time_alone_is_enough_and_the_calendar_model_answers(client):
    body = predict(client, datetime=ORDINARY)
    assert body["vehicles_per_hour"] > 3000 and body["model_used"] == "calendar only"
    assert body["holiday"] == {"name": None, "counts_as_holiday": False, "decided_by": "calendar"}
    assert body["typical_error"] > 0 and len(body["prediction_id"]) == 32


def test_with_weather_the_weather_model_answers(client):
    with_weather = predict(client, datetime=ORDINARY, weather=WEATHER)
    without = predict(client, datetime=ORDINARY)
    assert with_weather["model_used"] == "calendar and weather"
    assert with_weather["model_version"] != without["model_version"]
    assert with_weather["typical_error"] <= without["typical_error"]
    assert float(client.post("/api/v1/predict", json={"datetime": ORDINARY}).headers["X-Process-Time-ms"]) > 0


def test_the_holiday_is_worked_out_from_the_date(client):
    ordinary, holiday = predict(client, datetime=ORDINARY), predict(client, datetime=THANKSGIVING)
    assert holiday["holiday"]["name"] == "Thanksgiving Day" and holiday["holiday"]["counts_as_holiday"]
    assert holiday["vehicles_per_hour"] < 0.5 * ordinary["vehicles_per_hour"]
    veterans = predict(client, datetime="2026-11-11T08:00")  # named, but an ordinary commuting day
    assert veterans["holiday"]["name"] == "Veterans Day" and not veterans["holiday"]["counts_as_holiday"]
    assert veterans["vehicles_per_hour"] > 0.8 * ordinary["vehicles_per_hour"]


def test_the_holiday_can_be_overridden(client):
    forced = predict(client, datetime=ORDINARY, is_holiday=True)
    assert forced["holiday"]["decided_by"] == "request" and forced["holiday"]["counts_as_holiday"]
    assert forced["vehicles_per_hour"] < 0.5 * predict(client, datetime=ORDINARY)["vehicles_per_hour"]
    unforced = predict(client, datetime=THANKSGIVING, is_holiday=False)
    assert unforced["vehicles_per_hour"] > 3000


def test_rush_hour_is_busier_than_sunday_and_night(client):
    rush = predict(client, datetime=ORDINARY)["vehicles_per_hour"]
    sunday = predict(client, datetime="2026-11-22T08:00")["vehicles_per_hour"]
    night = predict(client, datetime="2026-11-19T03:00")["vehicles_per_hour"]
    assert rush > sunday > night >= 0


def test_day_forecast(client):
    r = client.post("/api/v1/predict/day", json={"date": "2026-11-19"})
    assert r.status_code == 200
    body = r.json()
    assert len(body["hours"]) == 24 and body["peak_vehicles"] == max(h["vehicles_per_hour"] for h in body["hours"])
    assert body["total_vehicles"] == sum(h["vehicles_per_hour"] for h in body["hours"]) and body["model_used"] == "calendar only"
    with_weather = client.post("/api/v1/predict/day", json={"date": "2026-11-19", "weather": WEATHER}).json()
    assert with_weather["model_used"] == "calendar and weather"


def test_batch_mixes_both_models_and_keeps_the_order(client):
    hours = [{"datetime": ORDINARY}, {"datetime": ORDINARY, "weather": WEATHER}, {"datetime": "2026-11-19T03:00"}]
    r = client.post("/api/v1/predict/batch", json={"hours": hours})
    assert r.status_code == 200
    got = r.json()["predictions"]
    assert [p["model_used"] for p in got] == ["calendar only", "calendar and weather", "calendar only"]
    assert got[0]["vehicles_per_hour"] == predict(client, datetime=ORDINARY)["vehicles_per_hour"]
    assert got[2]["vehicles_per_hour"] < got[0]["vehicles_per_hour"]


@pytest.mark.parametrize("body", [
    {"datetime": "tomorrow"},
    {"datetime": "2026-02-30T08:00"},
    {},
    {"datetime": ORDINARY, "year": 2026},
    {"datetime": ORDINARY, "temp_c": 4},                                       # weather belongs inside "weather"
    {"datetime": ORDINARY, "weather": {**WEATHER, "weather_main": "Tornado"}},
    {"datetime": ORDINARY, "weather": {**WEATHER, "temp_c": 80}},
    {"datetime": ORDINARY, "weather": {**WEATHER, "clouds_all": 101}},
    {"datetime": ORDINARY, "weather": {"temp_c": 4}},                          # incomplete weather
    {"datetime": ORDINARY, "is_holiday": "maybe"},
])
def test_bad_input_is_refused_with_an_explanation(client, body):
    r = client.post("/api/v1/predict", json=body)
    assert r.status_code == 422
    reply = r.json()
    assert reply["error"] == "invalid_request" and len(reply["message"]) > 20 and reply["details"]


def test_malformed_requests_do_not_crash_the_service(client):
    broken = client.post("/api/v1/predict", content=b'{"datetime": ', headers={"Content-Type": "application/json"})
    assert broken.status_code == 422 and "not valid JSON" in broken.json()["message"]
    text = client.post("/api/v1/predict", content=b"hello", headers={"Content-Type": "text/plain"})
    assert text.status_code == 422 and "JSON" in text.json()["message"]
    too_many = client.post("/api/v1/predict/batch", json={"hours": [{"datetime": ORDINARY}] * 169})
    assert too_many.status_code == 422
    assert client.post("/api/v1/predict/batch", json={"hours": []}).status_code == 422
    missing = client.get("/no-such-page")
    assert missing.status_code == 404 and missing.json()["error"] == "not_found"
    wrong = client.get("/api/v1/predict")
    assert wrong.status_code == 405 and wrong.json()["error"] == "method_not_allowed"
    assert client.get("/health").json()["status"] == "ok"


def test_feedback_loop(client):
    before = client.get("/api/v1/feedback/summary").json()["reports_matched"]
    p = predict(client, datetime=ORDINARY)
    ack = client.post("/api/v1/feedback", json={"prediction_id": p["prediction_id"],
                                                "actual_vehicles": p["vehicles_per_hour"] + 100}).json()
    assert ack["received"] and ack["matched"] and 99 <= ack["error"] <= 101
    unknown = client.post("/api/v1/feedback", json={"prediction_id": "not-a-real-id", "actual_vehicles": 5000}).json()
    assert unknown["received"] and not unknown["matched"]
    summary = client.get("/api/v1/feedback/summary").json()
    assert summary["reports_matched"] == before + 1 and summary["mae"] > 0
    assert client.post("/api/v1/feedback", json={"prediction_id": "x", "actual_vehicles": -1}).status_code == 422


def test_metrics_and_pages(client):
    stats = client.get("/api/v1/metrics").json()
    assert stats["predictions"] >= 1 and stats["rejected_requests"] >= 1 and stats["errors"] == 0
    page = client.get("/")
    assert page.status_code == 200 and "text/html" in page.headers["content-type"] and "Forecast this day" in page.text
    assert client.get("/docs").status_code == 200
    schema = client.get("/openapi.json").json()
    assert {"/api/v1/predict", "/api/v1/predict/day", "/api/v1/predict/batch", "/api/v1/feedback"} <= set(schema["paths"])
