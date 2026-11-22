"""Sanity checks on the raw data. Hard failures raise; everything else is reported."""

from __future__ import annotations

import pandas as pd

from .config import Config
from .rowlevel import RAINY, holiday_name

RAW_NUMERIC = ["temp", "rain_1h", "snow_1h", "clouds_all", "traffic_volume"]
RAW_TEXT = ["holiday", "weather_main", "weather_description", "date_time"]


def validate_raw(df: pd.DataFrame, cfg: Config) -> dict:
    """Check shape, columns and types; return a structural audit (no target statistics by split)."""
    cols = [c for c in df.columns if c != "row_id"]
    problems = []
    if len(df) != cfg.data.expected_rows:
        problems.append(f"expected {cfg.data.expected_rows} rows, found {len(df)}")
    if len(cols) != cfg.data.expected_columns:
        problems.append(f"expected {cfg.data.expected_columns} columns, found {len(cols)}")
    missing = sorted(set(RAW_NUMERIC + RAW_TEXT) - set(cols))
    if missing:
        problems.append(f"missing columns: {missing}")
    for c in RAW_NUMERIC:
        if c in df and not pd.api.types.is_numeric_dtype(df[c]):
            problems.append(f"{c} should be numeric")
    if problems:
        raise ValueError("raw data failed validation: " + "; ".join(problems))
    stamps = pd.to_datetime(df["date_time"])
    year = stamps.dt.year
    hourly = df.assign(year=year, stamp=stamps).drop_duplicates("date_time")
    low = hourly["traffic_volume"] < cfg.cleaning.min_plausible_volume
    regular = hourly[~low]
    quietest = regular.groupby([regular["stamp"].dt.dayofweek, regular["stamp"].dt.hour])["traffic_volume"].mean().min()
    rainy, snowy = df["weather_main"].isin(RAINY), df["weather_main"] == "Snow"
    amounts = [{"year": int(y), "rain_weather_rows": int(rainy[year == y].sum()),
                "of_which_rain_1h_recorded": int((rainy & (df["rain_1h"] > 0))[year == y].sum()),
                "snow_weather_rows": int(snowy[year == y].sum()),
                "of_which_snow_1h_recorded": int((snowy & (df["snow_1h"] > 0))[year == y].sum())}
               for y in sorted(year.unique())]
    days = pd.DataFrame({"day": stamps.dt.normalize(), "flag": df["holiday"] != "None"}).groupby("day")["flag"].any()
    unmarked = [str(d.date()) for d, flagged in days.items()
                if not flagged and holiday_name(d.year, d.month, d.day) != "None"]
    counts = stamps.value_counts()
    full = pd.date_range(stamps.min(), stamps.max(), freq="h")
    hours = pd.Series(sorted(counts.index))
    gaps = hours.diff().dt.total_seconds() / 3600
    return {
        "rows": len(df), "columns": len(cols),
        "missing_values": {k: int(v) for k, v in df[cols].isna().sum().items() if v},
        "missing_cells_total": int(df[cols].isna().sum().sum()),
        "exact_duplicate_rows": int(df.duplicated(subset=cols).sum()),
        "hours_whose_rows_disagree_on_volume": int((df.groupby("date_time")["traffic_volume"].nunique() > 1).sum()),
        "low_volume_hours": int(low.sum()),
        "low_volume_hours_by_year": {int(k): int(v) for k, v in low.groupby(hourly["year"]).sum().items()},
        "quietest_regular_hour_mean": float(quietest),
        "precipitation_amounts_by_year": amounts,
        "named_days_never_marked": unmarked,
        "first_hour": str(stamps.min()), "last_hour": str(stamps.max()),
        "distinct_hours": len(counts), "hours_with_several_rows": int((counts > 1).sum()),
        "max_rows_per_hour": int(counts.max()), "hours_in_range": len(full),
        "missing_hours": int(len(full) - len(counts)), "gaps_over_24h": int((gaps > 24).sum()),
        "largest_gap_hours": float(gaps.max()), "largest_gap_ends": str(hours[gaps.idxmax()]),
        "zero_kelvin_temps": int((df["temp"] < cfg.cleaning.min_valid_temp_kelvin).sum()),
        "rain_over_limit": int((df["rain_1h"] > cfg.cleaning.max_valid_rain_mm).sum()),
        "zero_volume_rows": int((df["traffic_volume"] == 0).sum()),
        "holiday_rows": int((df["holiday"] != "None").sum()),
        "rows_per_year": {int(k): int(v) for k, v in stamps.dt.year.value_counts().sort_index().items()},
    }
