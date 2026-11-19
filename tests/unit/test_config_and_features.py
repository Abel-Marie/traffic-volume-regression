import ast
import math
from pathlib import Path

import pandas as pd
import pytest
import yaml
from pandas.tseries.holiday import USFederalHolidayCalendar

from traffic_volume import rowlevel
from traffic_volume.config import apply_fast_mode, config_from_dict
from traffic_volume.rowlevel import (
    CATEGORICAL_FEATURES,
    FEATURE_INPUTS,
    RAW_VARIABLES,
    derive_features,
    features_using_only,
    holiday_name,
)

HOUR = {"hour": 8, "weekday": 1, "month": 1, "is_holiday": 0, "temp": 273.15, "rain_1h": 0.0, "snow_1h": 0.5,
        "clouds_all": 90, "weather_main": "Drizzle"}


def test_config_loads_and_rejects_typos(cfg, root):
    assert cfg.split.train_until_year < cfg.split.validation_year < cfg.split.test_year
    raw = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    raw["tuning"]["typo"] = 1
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict(raw)


def test_grid_must_match_budget(root):
    raw = yaml.safe_load((root / "configs" / "config.yaml").read_text())
    raw["tuning"]["lgbm_budget"] += 1
    with pytest.raises(ValueError, match="lgbm_budget"):
        config_from_dict(raw)


def test_fast_mode(cfg):
    assert apply_fast_mode(cfg).is_fast and apply_fast_mode(cfg).tuning.cv_folds < cfg.tuning.cv_folds


def test_features(cfg):
    f = derive_features(HOUR, cfg.features.sources)
    assert f["temp_c"] == pytest.approx(0.0)
    assert f["hour_sin"] == pytest.approx(math.sin(2 * math.pi * 8 / 24))
    assert f["rush_hour"] == 1.0 and f["is_weekend"] == 0.0 and f["hour_slot"] == "weekday-08"
    assert f["is_raining"] == 1.0 and f["is_snowing"] == 1.0 and f["weather_group"] == "Rain"
    assert f["month_cos"] == pytest.approx(1.0)
    sunday = derive_features({**HOUR, "weekday": 6}, cfg.features.sources)
    assert sunday["rush_hour"] == 0.0 and sunday["hour_slot"] == "weekend-08"


def test_every_feature_declares_its_inputs(cfg):
    produced = set(derive_features(HOUR, cfg.features.sources))
    assert produced == set(FEATURE_INPUTS)
    assert all(set(v) <= set(RAW_VARIABLES) for v in FEATURE_INPUTS.values())
    assert produced >= CATEGORICAL_FEATURES


def test_univariate_features_do_not_leak_other_variables():
    assert features_using_only(["hour"]) == ["hour", "hour_sin", "hour_cos"]
    assert "rush_hour" in features_using_only(["hour", "weekday"])
    assert features_using_only(["temp"]) == ["temp_c"]


def test_rowlevel_uses_only_the_standard_library():
    tree = ast.parse(Path(rowlevel.__file__).read_text())
    modules = {n.module if isinstance(n, ast.ImportFrom) else a.name
               for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
               for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert modules <= {"__future__", "math", "datetime"}


def test_holiday_calendar_matches_the_federal_calendar():
    """The calendar rules give the US federal holidays on their observed dates (Juneteenth, added in 2021,
    is not in the 2012 to 2018 data and is not one of the named days)."""
    federal = {d for d in USFederalHolidayCalendar().holidays(start="2000-01-01", end="2040-12-31")
               if not (d.month == 6 and d.day in (18, 19, 20))}
    mine = {d for d in pd.date_range("2000-01-01", "2040-12-31")
            if holiday_name(d.year, d.month, d.day) not in ("None", "State Fair")}
    assert mine == federal


def test_holiday_calendar_reproduces_every_day_the_source_marks(root):
    raw = pd.read_csv(root / "data" / "mirror" / "traffic.csv.gz", keep_default_na=False, na_values=[""])
    marked = raw[raw["holiday"] != "None"].drop_duplicates("date_time")
    stamps = pd.to_datetime(marked["date_time"])
    assert len(marked) > 50
    assert [holiday_name(t.year, t.month, t.day) for t in stamps] == marked["holiday"].tolist()


def test_named_days_examples():
    assert holiday_name(2017, 11, 23) == "Thanksgiving Day" and holiday_name(2017, 11, 24) == "None"
    assert holiday_name(2016, 12, 26) == "Christmas Day"      # 25 December 2016 was a Sunday
    assert holiday_name(2021, 12, 31) == "New Years Day"      # 1 January 2022 was a Saturday
    assert holiday_name(2018, 8, 23) == "State Fair" and holiday_name(2018, 9, 3) == "Labor Day"
