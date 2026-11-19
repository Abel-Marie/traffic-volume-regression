"""Replay real hours against a running API and measure latency and agreement."""

from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd


def to_api_payload(row: dict, weather: bool = True) -> dict:
    """One cleaned data row as the JSON body ``/api/v1/predict`` expects (temperature in Celsius).

    With ``weather=False`` the body carries the time alone, and the service answers with its
    calendar-only model. The holiday is never sent: the service works it out from the date."""
    body = {"datetime": pd.Timestamp(row["date_time"]).strftime("%Y-%m-%dT%H:00")}
    if weather:
        body["weather"] = {"temp_c": round(float(row["temp"]) - 273.15, 6), "rain_1h": float(row["rain_1h"]),
                           "snow_1h": float(row["snow_1h"]), "clouds_all": int(row["clouds_all"]),
                           "weather_main": row["weather_main"]}
    return body


def post_json(url: str, body: dict, timeout: float = 30.0) -> tuple[dict, float, float | None]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read())
        server = resp.headers.get("X-Process-Time-ms")
    return payload, (time.perf_counter() - start) * 1000, float(server) if server else None


def replay_requests(base_url: str, rows: pd.DataFrame, expected: np.ndarray | None = None) -> dict:
    """Send each hour as one request; the first request is reported separately (cold start)."""
    url = base_url.rstrip("/") + "/api/v1/predict"
    client, server, diffs, errors = [], [], [], 0
    for i, row in enumerate(rows.to_dict("records")):
        try:
            payload, ms, srv = post_json(url, to_api_payload(row))
        except Exception:  # noqa: BLE001 - count failures, keep measuring
            errors += 1
            continue
        client.append(ms)
        if srv is not None:
            server.append(srv)
        if expected is not None:
            diffs.append(abs(payload["raw_prediction"] - float(expected[i])))
    if not client:
        raise RuntimeError(f"no successful requests to {url}")
    warm = np.array(client[1:]) if len(client) > 1 else np.array(client)
    return {"url": url, "requests": len(rows), "errors": errors, "error_rate": errors / len(rows),
            "first_request_ms": float(client[0]), "warm_p50_ms": float(np.percentile(warm, 50)),
            "warm_p95_ms": float(np.percentile(warm, 95)), "warm_p99_ms": float(np.percentile(warm, 99)),
            "server_p50_ms": float(np.percentile(server, 50)) if server else None,
            "max_abs_diff_vs_offline": float(max(diffs)) if diffs else None,
            "client_latencies_ms": [float(v) for v in client]}
