"""I-94 traffic volume API (FastAPI). Vercel loads the ``app`` object from this file."""

from __future__ import annotations

import datetime as dt
import json
import logging
import math
import sys
import time
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np
from core.inference import TrafficModel
from core.schemas import (
    BatchRequest,
    BatchResponse,
    DayForecast,
    DayRequest,
    Feedback,
    FeedbackAck,
    HourRequest,
    PredictionResponse,
)
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

HERE = Path(__file__).resolve().parent
MODEL = TrafficModel()

log = logging.getLogger("api")
if not log.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

STARTED = time.time()
STATS = {"requests": 0, "errors": 0, "rejected": 0, "predictions": 0, "latencies_ms": deque(maxlen=2000)}
# Recent predictions, so that an observed count posted later can be compared with what was forecast.
# Held in memory: a serverless instance forgets them when it stops. The request log is the durable record.
RECENT: OrderedDict[str, dict] = OrderedDict()
RECENT_LIMIT = 20000
FEEDBACK: deque = deque(maxlen=20000)

app = FastAPI(
    title="I-94 traffic volume",
    version=MODEL.version,
    description=(
        "Forecasts westbound vehicles per hour on Interstate 94 between Minneapolis and St Paul (MnDOT station "
        "301) from the calendar and, when it is known, the weather.\n\n"
        "* Send only a time and the **calendar-only model** answers; the public holiday is worked out from the date.\n"
        "* Add the expected weather and the **calendar-and-weather model** answers.\n"
        "* Every reply says which model answered and its typical error on the 2018 test year.\n\n"
        "See `/api/v1/model` for what the models use, how they were tested and their limits."),
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


def problem(status: int, error: str, message: str, details: list | None = None) -> JSONResponse:
    """Every error has the same shape: a short code, a sentence a person can read, and details."""
    body = {"error": error, "message": message}
    if details:
        body["details"] = details
    return JSONResponse(status_code=status, content=body)


@app.exception_handler(RequestValidationError)
async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    details = []
    for e in exc.errors():
        where = ".".join(str(p) for p in e["loc"] if p != "body") or "request body"
        if e["type"] == "json_invalid":
            where, text = "request body", "it is not valid JSON"
        elif where == "request body":
            text = "send a JSON object with Content-Type: application/json"
        else:
            text = e["msg"]
        details.append({"field": where, "problem": text})
    first = "; ".join(f"{d['field']}: {d['problem']}" for d in details[:3])
    more = f" (and {len(details) - 3} more)" if len(details) > 3 else ""
    return problem(422, "invalid_request", f"The request was not understood. {first}{more}. See /docs for the "
                                           "expected fields.", details)


@app.exception_handler(StarletteHTTPException)
async def http_problem(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    messages = {404: "There is nothing at this address. The demo page is at /, the documentation at /docs.",
                405: "This address does not accept that kind of request. Predictions are POST requests; see /docs."}
    return problem(exc.status_code, {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error"),
                   messages.get(exc.status_code, str(exc.detail)))


@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception) -> JSONResponse:
    log.info(json.dumps({"event": "error", "path": request.url.path, "type": type(exc).__name__}))
    return problem(500, "internal_error", "Something went wrong on the server. The request was not processed.")


@app.middleware("http")
async def timing_and_logging(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - start) * 1000
        STATS["requests"] += 1
        STATS["errors"] += int(status >= 500)
        STATS["rejected"] += int(400 <= status < 500)
        if request.url.path.startswith("/api/"):
            STATS["latencies_ms"].append(ms)
        if status != 500:
            response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
        log.info(json.dumps({"path": request.url.path, "method": request.method, "status": status,
                             "ms": round(ms, 2), "model_version": MODEL.version}))


def remember(predictions: list[dict]) -> list[dict]:
    for p in predictions:
        RECENT[p["prediction_id"]] = {"predicted": p["raw_prediction"], "datetime": p["datetime"].isoformat(),
                                      "model_used": p["model_used"]}
    while len(RECENT) > RECENT_LIMIT:
        RECENT.popitem(last=False)
    STATS["predictions"] += len(predictions)
    return predictions


def item(moment: dt.datetime, conditions) -> dict:
    return {"moment": moment, "is_holiday": conditions.is_holiday,
            "weather": conditions.weather.raw() if conditions.weather else None}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(HERE / "public" / "index.html")


@app.get("/health", summary="Is the service up, and which model is it serving?")
def health() -> dict:
    return {"status": "ok", "model_version": MODEL.version,
            "calendar_model_version": MODEL.calendar.version if MODEL.calendar else None,
            "uptime_s": round(time.time() - STARTED, 1)}


@app.get("/api/v1/model", summary="Model card: inputs, test results and limits")
def model_card() -> dict:
    m = MODEL.metadata
    fallback = m.get("fallback")
    return {
        "task": m["task"], "trained_on": m["trained_on"], "trained_utc": m["trained_utc"], "estimator": m["estimator"],
        "params": m["params"],
        "models": {
            "calendar and weather" if MODEL.needs_weather else "calendar only": {
                "version": m["model_version"], "inputs": m["input_variables"], "features": m["input_order"],
                "test_metrics": m["test_metrics"], "used_when": "the request includes the weather"
                if MODEL.needs_weather else "always"},
            **({"calendar only": {"version": fallback["model_version"], "inputs": fallback["input_variables"],
                                  "features": fallback["input_order"], "test_metrics": fallback["test_metrics"],
                                  "used_when": "the request has no weather"}} if fallback else {})},
        "test_metrics": m["test_metrics"], "validation_metrics": m.get("validation_metrics"),
        "test_reference": m.get("test_reference", {}),
        "test_year": "2018, January to September; scored twice (see first_evaluation)",
        "first_evaluation": m.get("first_evaluation"),
        "quiet_holidays": m["quiet_holidays"], "inputs_note": m["inputs_note"], "not_used": m["not_used"],
        "limits": ["One counting station and one direction, 2012 to 2018.",
                   "Incidents, closures, heavy snow and unusual days such as the day after Thanksgiving are not foreseen.",
                   "The weather it was trained on was observed; a forecast is less exact.",
                   "No trend: a lasting change to the road needs retraining."],
        "data": m["data"],
    }


@app.post("/api/v1/predict", response_model=PredictionResponse, summary="Forecast one hour")
def predict(hour: HourRequest) -> dict:
    start = time.perf_counter()
    result = remember(MODEL.predict([item(hour.datetime, hour)]))[0]
    return {**result, "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


@app.post("/api/v1/predict/batch", response_model=BatchResponse, summary="Forecast up to a week of hours")
def predict_batch(batch: BatchRequest) -> dict:
    start = time.perf_counter()
    results = remember(MODEL.predict([item(h.datetime, h) for h in batch.hours]))
    return {"predictions": results, "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


@app.post("/api/v1/predict/day", response_model=DayForecast, summary="Forecast the 24 hours of a day")
def predict_day(day: DayRequest) -> dict:
    start = time.perf_counter()
    moments = [dt.datetime.combine(day.date, dt.time(h)) for h in range(24)]
    results = remember(MODEL.predict([item(m, day) for m in moments]))
    values = [r["vehicles_per_hour"] for r in results]
    peak = int(np.argmax(values))
    first = results[0]
    return {"date": day.date, "hours": [{"hour": h, "vehicles_per_hour": v} for h, v in enumerate(values)],
            "peak_hour": peak, "peak_vehicles": values[peak], "total_vehicles": int(sum(values)),
            "model_used": first["model_used"], "model_version": first["model_version"],
            "typical_error": first["typical_error"], "holiday": first["holiday"],
            "latency_ms": round((time.perf_counter() - start) * 1000, 3)}


@app.post("/api/v1/feedback", response_model=FeedbackAck, summary="Report the count that was really observed")
def feedback(report: Feedback) -> dict:
    known = RECENT.get(report.prediction_id)
    error = None
    if known:
        error = int(round(report.actual_vehicles - known["predicted"]))
        FEEDBACK.append({"error": error, "actual": report.actual_vehicles, "model_used": known["model_used"]})
    log.info(json.dumps({"event": "feedback", "prediction_id": report.prediction_id,
                         "actual_vehicles": report.actual_vehicles, "error": error, "model_version": MODEL.version}))
    note = ("Recorded and compared with the forecast." if known else
            "Recorded in the log. This instance no longer remembers that prediction, so it is not in the summary.")
    return {"received": True, "matched": known is not None, "error": error, "note": note}


@app.get("/api/v1/feedback/summary", summary="Running error on the counts reported so far")
def feedback_summary() -> dict:
    errors = np.array([f["error"] for f in FEEDBACK], dtype=float)
    out = {"storage": "memory of this instance only; the request log is the durable record",
           "reports_matched": int(len(errors))}
    if len(errors):
        out.update({"mae": round(float(np.mean(np.abs(errors))), 1), "rmse": round(float(math.sqrt(np.mean(errors ** 2))), 1),
                    "mean_error": round(float(np.mean(errors)), 1),
                    "note": "mean_error is observed minus predicted: positive means the model forecasts too few."})
    return out


@app.get("/api/v1/metrics", summary="Requests and latency of this instance")
def metrics() -> dict:
    lat = np.asarray(STATS["latencies_ms"]) if STATS["latencies_ms"] else np.asarray([0.0])
    return {"scope": "this instance only (serverless instances are stateless)",
            "uptime_s": round(time.time() - STARTED, 1), "requests": STATS["requests"], "errors": STATS["errors"],
            "rejected_requests": STATS["rejected"], "predictions": STATS["predictions"],
            "api_latency_ms": {"p50": round(float(np.percentile(lat, 50)), 2),
                               "p95": round(float(np.percentile(lat, 95)), 2),
                               "p99": round(float(np.percentile(lat, 99)), 2)}}
