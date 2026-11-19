"""Request and response models. Ranges follow the training data (Minneapolis weather, 2012 to 2018)."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

WeatherName = Literal["Clear", "Clouds", "Rain", "Drizzle", "Snow", "Thunderstorm", "Squall", "Fog", "Mist", "Haze",
                      "Smoke"]

EXAMPLE_WEATHER = {"temp_c": 4.0, "weather_main": "Clouds", "clouds_all": 75, "rain_1h": 0.0, "snow_1h": 0.0}
EXAMPLE_HOUR = {"datetime": "2026-11-19T08:00", "weather": EXAMPLE_WEATHER}


class Weather(BaseModel):
    """The weather expected in the hour (a forecast, when planning ahead). Optional as a whole: leave it
    out and the calendar-only model answers."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE_WEATHER})
    temp_c: float = Field(ge=-45, le=45, description="Air temperature, degrees Celsius")
    weather_main: WeatherName = Field(description="Kind of weather")
    clouds_all: int = Field(ge=0, le=100, description="Cloud cover, %")
    rain_1h: float = Field(default=0.0, ge=0, le=100, description="Rain in the hour, mm")
    snow_1h: float = Field(default=0.0, ge=0, le=10, description="Snow in the hour, mm")

    def raw(self) -> dict:
        """The raw-variable form the shared feature code expects (temperature in Kelvin, as in the data)."""
        return {"temp": self.temp_c + 273.15, "rain_1h": self.rain_1h, "snow_1h": self.snow_1h,
                "clouds_all": self.clouds_all, "weather_main": self.weather_main}


class Conditions(BaseModel):
    model_config = ConfigDict(extra="forbid")
    is_holiday: bool | None = Field(
        default=None, description="Leave out to let the service work the holiday out from the date. Set true or "
                                  "false only to override it (true means a day on which most workplaces close).")
    weather: Weather | None = Field(
        default=None, description="Weather expected in the hour. Leave out when it is not known: the "
                                  "calendar-only model then answers.")


class HourRequest(Conditions):
    """One hour, in local time (US Central), with the weather if it is known."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        EXAMPLE_HOUR, {"datetime": "2026-11-26T08:00"}]})
    datetime: dt.datetime = Field(description="The hour to forecast, local time, for example 2026-11-19T08:00")


class DayRequest(Conditions):
    """A whole day; the same weather is used for every hour."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"date": "2026-11-19"}, {"date": "2026-11-19", "weather": EXAMPLE_WEATHER}]})
    date: dt.date = Field(description="The day to forecast, for example 2026-11-19")


class Holiday(BaseModel):
    name: str | None = Field(description="The named day on that date, if any")
    counts_as_holiday: bool = Field(description="Whether the model treated the day as a quiet holiday")
    decided_by: Literal["calendar", "request"]


class Prediction(BaseModel):
    prediction_id: str = Field(description="Quote this id when reporting the count that was really observed")
    datetime: dt.datetime
    vehicles_per_hour: int = Field(description="Forecast of westbound vehicles in the hour (never below zero)")
    raw_prediction: float = Field(description="Unrounded model output, for monitoring")
    model_used: Literal["calendar and weather", "calendar only"]
    model_version: str
    typical_error: int = Field(description="RMSE of the model that answered on the 2018 test year, vehicles per hour")
    holiday: Holiday


class PredictionResponse(Prediction):
    latency_ms: float


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    hours: list[HourRequest] = Field(min_length=1, max_length=168, description="1 to 168 hours (one week)")


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    latency_ms: float


class HourValue(BaseModel):
    hour: int
    vehicles_per_hour: int


class DayForecast(BaseModel):
    date: dt.date
    hours: list[HourValue]
    peak_hour: int
    peak_vehicles: int
    total_vehicles: int
    model_used: Literal["calendar and weather", "calendar only"]
    model_version: str
    typical_error: int
    holiday: Holiday
    latency_ms: float


class Feedback(BaseModel):
    """The count that was really observed for an earlier prediction."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "prediction_id": "3f1c9a6e0d2b4c7f8a1e5b9d2c4f6a80", "actual_vehicles": 5120}})
    prediction_id: str = Field(min_length=1, max_length=64)
    actual_vehicles: int = Field(ge=0, le=20000, description="Vehicles counted in that hour")


class FeedbackAck(BaseModel):
    received: bool
    matched: bool = Field(description="Whether this instance still remembers the prediction")
    error: int | None = Field(description="Observed minus predicted, when matched")
    note: str
