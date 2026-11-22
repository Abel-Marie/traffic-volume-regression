"""Cleaning rules, each logged with the rows it removed or changed.

Only fixed, documented rules: no threshold is estimated from the data, so
cleaning runs safely before the chronological split.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .rowlevel import holiday_name

SOURCE_COLUMNS = ["holiday", "temp", "rain_1h", "snow_1h", "clouds_all", "weather_main", "weather_description",
                  "date_time", "traffic_volume"]


def collapse_hours(df: pd.DataFrame, severity: list[str]) -> pd.DataFrame:
    """One row per hour. Volume is identical across an hour's rows; weather keeps the worst case."""
    rank = {w: i for i, w in enumerate(severity)}
    work = df.assign(_rank=df["weather_main"].map(rank).fillna(len(rank)))
    worst = work.sort_values(["date_time", "_rank"]).groupby("date_time", as_index=False).first()
    agg = df.groupby("date_time").agg(temp=("temp", "mean"), rain_1h=("rain_1h", "max"), snow_1h=("snow_1h", "max"),
                                      clouds_all=("clouds_all", "max"), traffic_volume=("traffic_volume", "first"),
                                      row_id=("row_id", "min"), weather_rows=("row_id", "size"))
    keep = worst[["date_time", "weather_main", "weather_description", "holiday"]].set_index("date_time")
    return agg.join(keep).reset_index()


def calendar_holiday_names(stamps: pd.Series) -> pd.Series:
    """The named day for each timestamp, from the calendar rules (not from the data)."""
    days = stamps.dt.normalize()
    lookup = {d: holiday_name(d.year, d.month, d.day) for d in days.unique()}
    return days.map(lookup)


def clean_traffic(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, list[dict]]:
    """Apply the cleaning rules in order and return one row per recorded hour, plus a log."""
    c = cfg.cleaning
    out = df.copy()
    log: list[dict] = []

    def record(rule: str, before: int, why: str, changed: int | None = None) -> None:
        log.append({"rule": rule, "rows_before": before, "rows_after": len(out),
                    "rows_removed": before - len(out), "values_changed": changed, "why": why})

    before = len(out)
    out = out[~out.duplicated(subset=SOURCE_COLUMNS)]
    record("drop_exact_duplicate_rows", before, "The same record stored twice.")

    before = len(out)
    out = out[out["temp"] >= c.min_valid_temp_kelvin]
    record("drop_zero_kelvin_temperature", before, "0 K is a sensor error, not weather.")

    before = len(out)
    out = out[out["rain_1h"] <= c.max_valid_rain_mm]
    record("drop_impossible_rainfall", before, f"More than {c.max_valid_rain_mm:.0f} mm in one hour is a sensor error.")

    before = len(out)
    out = out[out["traffic_volume"] >= c.min_plausible_volume]
    record("drop_closure_or_outage_hours", before,
           f"Fewer than {c.min_plausible_volume} vehicles in an hour: the road was closed or the counter was down.")

    stamp = pd.to_datetime(out["date_time"])
    day = stamp.dt.strftime("%Y-%m-%d")
    names = out.loc[out["holiday"] != "None"].assign(day=day).groupby("day")["holiday"].first()
    before_flags = int((out["holiday"] != "None").sum())
    out = out.assign(holiday=day.map(names).fillna("None").to_numpy())
    record("spread_holiday_to_whole_day", len(out),
           "The source marks a holiday only on its midnight row; every hour of that day is a holiday.",
           int((out["holiday"] != "None").sum()) - before_flags)

    by_calendar = calendar_holiday_names(stamp).to_numpy()
    current = out["holiday"].to_numpy()
    missed = (current == "None") & (by_calendar != "None")
    out = out.assign(holiday=[c_name if m else h for h, c_name, m in zip(current, by_calendar, missed)])
    record("name_holidays_with_no_midnight_row", len(out),
           "A holiday whose midnight hour is missing from the data was never marked; the calendar names it.",
           int(missed.sum()))

    before = len(out)
    out = collapse_hours(out, c.weather_severity)
    record("one_row_per_hour", before,
           "Hours with several weather rows repeat the same volume; keep one row, with the most severe weather.")

    out["date_time"] = pd.to_datetime(out["date_time"])
    out["quiet_holiday"] = out["holiday"].isin(c.quiet_holidays)
    named = int((out["holiday"] != "None").sum())
    record("treat_only_quiet_holidays_as_holidays", len(out),
           "Named days on which traffic stays near normal (Columbus Day, Veterans Day and others) are ordinary "
           "days for the model; only the quiet holidays listed in the configuration count.",
           named - int(out["quiet_holiday"].sum()))
    return out.sort_values("date_time").reset_index(drop=True), log


def parse_calendar(clean: pd.DataFrame) -> pd.DataFrame:
    """The raw variables the feature code and the API use: calendar parts plus the weather columns."""
    t = clean["date_time"]
    return pd.DataFrame({
        "hour": t.dt.hour, "weekday": t.dt.dayofweek, "month": t.dt.month,
        "is_holiday": clean["quiet_holiday"].astype(int),
        "temp": clean["temp"], "rain_1h": clean["rain_1h"], "snow_1h": clean["snow_1h"],
        "clouds_all": clean["clouds_all"], "weather_main": clean["weather_main"],
    }, index=clean.index)


def typical_weather(clean_rows: pd.DataFrame) -> dict:
    """The usual weather of each month in the given rows: median temperature and cloud cover, the most
    common weather label, no rain and no snow. It stands in for the weather when none is known, as when
    planning further ahead than any forecast reaches."""
    out = {}
    for month, part in clean_rows.groupby(clean_rows["date_time"].dt.month):
        out[int(month)] = {"temp": float(part["temp"].median()), "rain_1h": 0.0, "snow_1h": 0.0,
                           "clouds_all": float(part["clouds_all"].median()),
                           "weather_main": str(part["weather_main"].mode().iloc[0])}
    return out


def with_typical_weather(raw_vars: pd.DataFrame, typical: dict) -> pd.DataFrame:
    """The raw variables with every weather column replaced by the month's typical value."""
    out = raw_vars.copy()
    for column in ("temp", "rain_1h", "snow_1h", "clouds_all", "weather_main"):
        out[column] = [typical[int(m)][column] for m in out["month"]]
    return out


def holiday_evidence(clean_train: pd.DataFrame, target: str) -> pd.DataFrame:
    """For each named day: its traffic as a share of an ordinary day of the same weekday.

    Computed on the training years only. This is the evidence behind the fixed list of quiet
    holidays in the configuration; the list is not re-estimated when the pipeline runs.
    """
    t = clean_train.assign(wd=clean_train["date_time"].dt.dayofweek, h=clean_train["date_time"].dt.hour,
                           day=clean_train["date_time"].dt.normalize())
    normal = t[t["holiday"] == "None"].groupby(["wd", "h"])[target].mean().rename("normal")
    named = t[t["holiday"] != "None"].join(normal, on=["wd", "h"])
    per_day = named.groupby(["holiday", "day"]).agg(actual=(target, "sum"), normal=("normal", "sum"))
    per_day["share"] = per_day["actual"] / per_day["normal"]
    out = per_day.groupby("holiday")["share"].agg(days="size", share_of_normal="mean", lowest="min", highest="max")
    return out.sort_values("share_of_normal").reset_index()
