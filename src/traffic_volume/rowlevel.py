"""Per-row feature derivations in pure Python (standard library only).

This file is the single source of truth for feature logic. The training
pipeline applies it to every hour, and ``scripts/sync_service.py`` copies it
unchanged into ``service/core/features.py`` so the API computes exactly the
same features. Do not import numpy, pandas or scikit-learn here.

Input: one hour's raw variables (RAW_VARIABLES). ``FEATURE_INPUTS`` records
which raw variables each feature uses; the feature study relies on it to
build models from one, two or all variables without leaking others in.

``is_holiday`` means a *quiet* holiday: one of the six days on which traffic
really falls (the list is in the configuration and in the model's metadata).
``holiday_name`` gives the named day for any date, so cleaning can repair days
the source failed to mark and the API can work the holiday out from the date.
"""

from __future__ import annotations

import math
from datetime import date, timedelta

RAW_VARIABLES = ["hour", "weekday", "month", "is_holiday", "temp", "rain_1h", "snow_1h", "clouds_all",
                 "weather_main"]

CATEGORICAL_FEATURES = frozenset({"hour_slot", "weather_group"})

WEATHER_GROUPS = {
    "Clear": "Clear", "Clouds": "Clouds", "Rain": "Rain", "Drizzle": "Rain", "Snow": "Snow",
    "Thunderstorm": "Storm", "Squall": "Storm", "Fog": "Low visibility", "Mist": "Low visibility",
    "Haze": "Low visibility", "Smoke": "Low visibility",
}
RAINY = frozenset({"Rain", "Drizzle", "Thunderstorm", "Squall"})

FEATURE_INPUTS = {
    "hour": ["hour"], "hour_sin": ["hour"], "hour_cos": ["hour"],
    "weekday": ["weekday"], "is_weekend": ["weekday"],
    "rush_hour": ["hour", "weekday"], "hour_slot": ["hour", "weekday"],
    "month": ["month"], "month_sin": ["month"], "month_cos": ["month"],
    "is_holiday": ["is_holiday"],
    "temp_c": ["temp"],
    "rain_1h": ["rain_1h"], "is_raining": ["rain_1h", "weather_main"],
    "snow_1h": ["snow_1h"], "is_snowing": ["snow_1h", "weather_main"],
    "clouds_all": ["clouds_all"], "weather_group": ["weather_main"],
}


def nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The n-th given weekday of a month (Monday is 0)."""
    first = date(year, month, 1)
    return first + timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def last_weekday(year: int, month: int, weekday: int) -> date:
    last = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return last - timedelta(days=(last.weekday() - weekday) % 7)


def observed_day(day: date) -> date:
    """US federal rule for fixed-date holidays: a Saturday is observed on Friday, a Sunday on Monday."""
    return day + timedelta(days={5: -1, 6: 1}.get(day.weekday(), 0))


def named_days(year: int) -> dict:
    """The eleven days the source data names, for one year: ten US federal holidays on their observed
    dates and the opening day of the Minnesota State Fair (eleven days before Labor Day)."""
    labor = nth_weekday(year, 9, 0, 1)
    return {
        observed_day(date(year, 1, 1)): "New Years Day",
        nth_weekday(year, 1, 0, 3): "Martin Luther King Jr Day",
        nth_weekday(year, 2, 0, 3): "Washingtons Birthday",
        last_weekday(year, 5, 0): "Memorial Day",
        observed_day(date(year, 7, 4)): "Independence Day",
        labor - timedelta(days=11): "State Fair",
        labor: "Labor Day",
        nth_weekday(year, 10, 0, 2): "Columbus Day",
        observed_day(date(year, 11, 11)): "Veterans Day",
        nth_weekday(year, 11, 3, 4): "Thanksgiving Day",
        observed_day(date(year, 12, 25)): "Christmas Day",
    }


def holiday_name(year: int, month: int, day: int) -> str:
    """The named day on this date, in the source data's spelling; "None" otherwise."""
    wanted = date(year, month, day)
    for y in (year, year + 1):  # New Year's Day is observed on 31 December when 1 January is a Saturday
        name = named_days(y).get(wanted)
        if name:
            return name
    return "None"


def calendar_features(r: dict) -> dict:
    """Hour, weekday and month, as numbers, cycles and the hour-by-day-type profile."""
    hour, weekday, month = int(r["hour"]), int(r["weekday"]), int(r["month"])
    weekend = weekday >= 5
    h, m = 2 * math.pi * hour / 24, 2 * math.pi * (month - 1) / 12
    return {
        "hour": float(hour), "hour_sin": math.sin(h), "hour_cos": math.cos(h),
        "weekday": float(weekday), "is_weekend": 1.0 if weekend else 0.0,
        "rush_hour": 1.0 if not weekend and (6 <= hour <= 9 or 15 <= hour <= 18) else 0.0,
        "hour_slot": f"{'weekend' if weekend else 'weekday'}-{hour:02d}",
        "month": float(month), "month_sin": math.sin(m), "month_cos": math.cos(m),
    }


def holiday_features(r: dict) -> dict:
    """1 on a quiet holiday (the caller decides which days those are), else 0."""
    return {"is_holiday": float(r["is_holiday"])}


def weather_features(r: dict) -> dict:
    """Temperature in Celsius, precipitation amounts and flags, cloud cover and a weather group."""
    weather = str(r["weather_main"])
    rain, snow = float(r["rain_1h"]), float(r["snow_1h"])
    return {
        "temp_c": float(r["temp"]) - 273.15,
        "rain_1h": rain, "is_raining": 1.0 if rain > 0 or weather in RAINY else 0.0,
        "snow_1h": snow, "is_snowing": 1.0 if snow > 0 or weather == "Snow" else 0.0,
        "clouds_all": float(r["clouds_all"]),
        "weather_group": WEATHER_GROUPS.get(weather, "Other"),
    }


FEATURE_SOURCES = {"calendar": calendar_features, "holiday": holiday_features, "weather": weather_features}


def derive_features(r: dict, sources: list[str]) -> dict:
    """All features from the given groups, in a stable order."""
    out: dict = {}
    for source in sources:
        out.update(FEATURE_SOURCES[source](r))
    return out


def features_using_only(variables: list[str]) -> list[str]:
    """Features computed from these raw variables alone (no others leak in)."""
    allowed = set(variables)
    return [f for f, inputs in FEATURE_INPUTS.items() if set(inputs) <= allowed]
