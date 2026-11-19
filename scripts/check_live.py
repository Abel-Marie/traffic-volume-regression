"""Check a running copy of the API from the outside, as an examiner would.

    python scripts/check_live.py http://127.0.0.1:8765
    python scripts/check_live.py https://<your-app>.vercel.app

Checks the demo page, the documentation, health, the model card, single-hour, whole-day and batch
predictions, agreement with the exported models on real validation hours, the holiday logic, the
feedback loop and a set of deliberately bad requests. Writes reports/metrics/pipeline/live_check.json
and exits with a non-zero status if any check fails. Needs a finished pipeline run (for the exported
models).
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from traffic_volume.cleaning import parse_calendar  # noqa: E402
from traffic_volume.export import served_predictions  # noqa: E402
from traffic_volume.monitoring.latency import to_api_payload  # noqa: E402

ORDINARY, THANKSGIVING, VETERANS = "2026-11-19T08:00", "2026-11-26T08:00", "2026-11-11T08:00"
WEATHER = {"temp_c": 4.0, "weather_main": "Clouds", "clouds_all": 75}


def call(base: str, path: str, body=None, raw: bytes | None = None, content_type: str = "application/json",
         method: str | None = None) -> tuple[int, object, float]:
    """Status code, parsed body (JSON when possible, else text) and seconds taken. HTTP errors are returned."""
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method=method or ("GET" if data is None else "POST"),
                                 headers={"Content-Type": content_type, "User-Agent": "live-check/1.0"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            status, payload = resp.status, resp.read()
    except urllib.error.HTTPError as err:
        status, payload = err.code, err.read()
    seconds = time.perf_counter() - start
    try:
        return status, json.loads(payload), seconds
    except ValueError:
        return status, payload.decode("utf-8", "replace"), seconds


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    base = sys.argv[1].rstrip("/")
    export = ROOT / "artifacts" / "export"
    metadata = json.loads((export / "metadata.json").read_text(encoding="utf-8"))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")

    status, health, first_seconds = call(base, "/health")
    healthy = status == 200 and isinstance(health, dict) and health.get("status") == "ok"
    check("health (first request)", healthy, f"GET /health -> {status} in {first_seconds * 1000:.0f} ms")
    version = health.get("model_version") if isinstance(health, dict) else None
    check("serves the exported champion", version == metadata["model_version"],
          f"live {version}, exported {metadata['model_version']}")
    if metadata["fallback"]:
        live_calendar = health.get("calendar_model_version") if isinstance(health, dict) else None
        check("serves the exported calendar-only model", live_calendar == metadata["fallback"]["model_version"],
              f"live {live_calendar}, exported {metadata['fallback']['model_version']}")
    status, page, _ = call(base, "/")
    check("demo page loads", status == 200 and "Forecast this day" in str(page), f"GET / -> {status}")
    status, docs, _ = call(base, "/docs")
    check("API documentation loads", status == 200 and "swagger" in str(docs).lower(), f"GET /docs -> {status}")
    status, card, _ = call(base, "/api/v1/model")
    check("model card", status == 200 and card["test_metrics"] == metadata["test_metrics"]
          and card["quiet_holidays"] == metadata["quiet_holidays"], "test metrics and quiet holidays match the export")

    status, plain, _ = call(base, "/api/v1/predict", {"datetime": ORDINARY})
    check("prediction from a time alone", status == 200 and plain["vehicles_per_hour"] > 3000
          and plain["model_used"] == "calendar only", f"{ORDINARY} -> {plain.get('vehicles_per_hour')} vehicles, {plain.get('model_used')}")
    status, wet, _ = call(base, "/api/v1/predict", {"datetime": ORDINARY, "weather": WEATHER})
    check("prediction with weather uses the weather model", status == 200 and wet["model_used"] == "calendar and weather",
          f"{wet.get('vehicles_per_hour')} vehicles, {wet.get('model_used')}")
    status, holiday, _ = call(base, "/api/v1/predict", {"datetime": THANKSGIVING})
    check("a quiet holiday is recognised from the date", status == 200 and holiday["holiday"]["name"] == "Thanksgiving Day"
          and holiday["vehicles_per_hour"] < 0.5 * plain["vehicles_per_hour"],
          f"Thanksgiving 08:00 -> {holiday.get('vehicles_per_hour')} vehicles against {plain.get('vehicles_per_hour')} on an ordinary Thursday")
    status, veterans, _ = call(base, "/api/v1/predict", {"datetime": VETERANS})
    check("a named day that is not quiet stays an ordinary day", status == 200 and not veterans["holiday"]["counts_as_holiday"]
          and veterans["vehicles_per_hour"] > 0.8 * plain["vehicles_per_hour"], f"Veterans Day 08:00 -> {veterans.get('vehicles_per_hour')} vehicles")
    status, day, _ = call(base, "/api/v1/predict/day", {"date": "2026-11-19"})
    check("whole-day forecast", status == 200 and len(day["hours"]) == 24
          and day["peak_vehicles"] == max(h["vehicles_per_hour"] for h in day["hours"]),
          f"peak {day.get('peak_vehicles')} at {day.get('peak_hour')}:00")

    rows = pd.read_pickle(export / "parity_rows.pkl").head(100)
    raw = parse_calendar(rows)
    for label, weather, block in (("champion", True, metadata), ("calendar-only model", False, metadata["fallback"])):
        if block is None:
            continue
        offline = served_predictions(raw, block, export / block["file"])
        status, batch, _ = call(base, "/api/v1/predict/batch",
                                {"hours": [to_api_payload(r, weather=weather) for r in rows.to_dict("records")]})
        live = np.array([p["raw_prediction"] for p in batch["predictions"]]) if status == 200 else np.array([])
        diff = float(np.max(np.abs(live - offline))) if len(live) == len(offline) else float("nan")
        check(f"batch of 100 real hours agrees with the exported {label}", status == 200 and diff <= 0.01,
              f"largest difference {diff:.4f} vehicles per hour (limit 0.01)")

    status, ack, _ = call(base, "/api/v1/feedback", {"prediction_id": plain.get("prediction_id", "x"),
                                                     "actual_vehicles": int(plain.get("vehicles_per_hour", 0)) + 50})
    check("feedback accepted and matched", status == 200 and ack.get("received") and ack.get("matched"),
          f"error reported back: {ack.get('error') if isinstance(ack, dict) else ack}")
    status, summary, _ = call(base, "/api/v1/feedback/summary")
    check("feedback summary", status == 200 and summary.get("reports_matched", 0) >= 1,
          f"{summary.get('reports_matched') if isinstance(summary, dict) else '?'} report(s); MAE {summary.get('mae') if isinstance(summary, dict) else '?'}")
    status, stats, _ = call(base, "/api/v1/metrics")
    check("metrics", status == 200 and stats["predictions"] >= 1, f"{stats.get('requests')} requests, p95 {stats.get('api_latency_ms', {}).get('p95')} ms")

    bad = [
        ("malformed JSON", {"raw": b'{"datetime": '}, 422),
        ("empty body", {"raw": b""}, 422),
        ("plain text instead of JSON", {"raw": b"tomorrow at eight", "content_type": "text/plain"}, 422),
        ("no time", {"body": {}}, 422),
        ("an impossible date", {"body": {"datetime": "2026-02-30T08:00"}}, 422),
        ("words where a time belongs", {"body": {"datetime": "tomorrow"}}, 422),
        ("an unknown field", {"body": {"datetime": ORDINARY, "speed": 90}}, 422),
        ("weather outside 'weather'", {"body": {"datetime": ORDINARY, "temp_c": 4}}, 422),
        ("an impossible temperature", {"body": {"datetime": ORDINARY, "weather": {**WEATHER, "temp_c": 80}}}, 422),
        ("an unknown kind of weather", {"body": {"datetime": ORDINARY, "weather": {**WEATHER, "weather_main": "Tornado"}}}, 422),
        ("incomplete weather", {"body": {"datetime": ORDINARY, "weather": {"temp_c": 4}}}, 422),
    ]
    for name, kwargs, expected in bad:
        status, reply, _ = call(base, "/api/v1/predict", **kwargs)
        message = reply.get("message", "") if isinstance(reply, dict) else ""
        check(f"bad input: {name}", status == expected and len(message) > 20, f"{status}: {message[:110]}")
    status, reply, _ = call(base, "/api/v1/predict/batch", {"hours": [{"datetime": ORDINARY}] * 169})
    check("bad input: batch of 169 hours", status == 422, f"{status}")
    status, reply, _ = call(base, "/api/v1/feedback", {"prediction_id": "x", "actual_vehicles": -5})
    check("bad input: negative count in feedback", status == 422, f"{status}")
    status, reply, _ = call(base, "/no-such-page")
    check("unknown address", status == 404 and isinstance(reply, dict) and bool(reply.get("message")), f"{status}")
    status, reply, _ = call(base, "/api/v1/predict")
    check("wrong method", status == 405 and isinstance(reply, dict), f"{status}")
    status, health, _ = call(base, "/health")
    check("still healthy after the bad requests", status == 200, f"{status}")

    failed = [c for c in checks if not c["ok"]]
    result = {"url": base, "checked_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"), "model_version": version,
              "first_request_ms": round(first_seconds * 1000, 1), "checks": checks,
              "passed": len(checks) - len(failed), "failed": len(failed)}
    out = ROOT / "reports" / "metrics" / "pipeline" / "live_check.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n{result['passed']} of {len(checks)} checks passed. Wrote {out.relative_to(ROOT).as_posix()}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
