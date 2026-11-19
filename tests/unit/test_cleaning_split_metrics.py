import numpy as np
import pandas as pd
import pytest

from traffic_volume.cleaning import clean_traffic, holiday_evidence, parse_calendar
from traffic_volume.evaluation import (
    block_bootstrap_ci,
    fit_verdict,
    paired_block_bootstrap,
    regression_metrics,
    slice_report,
)
from traffic_volume.experiments import lag_frame
from traffic_volume.models import HolidayAwareBaseline, HourOfWeekBaseline
from traffic_volume.monitoring.drift import population_stability_index
from traffic_volume.split import chronological_split


def test_cleaning_rules(sample_raw, cfg):
    df = sample_raw.copy()
    df.loc[0, "temp"] = 0.0
    df.loc[1, "rain_1h"] = 9000.0
    df.loc[2, "traffic_volume"] = 0
    df.loc[3, "traffic_volume"] = cfg.cleaning.min_plausible_volume - 1
    stamp = df.loc[5, "date_time"]
    extra = df.iloc[[5]].assign(weather_main="Thunderstorm", row_id=10**6)
    copy = df.iloc[[7]].assign(row_id=10**6 + 1)  # the same record stored twice
    holiday_day = df.loc[20, "date_time"][:10]
    midnight = df.index[df["date_time"].str.startswith(holiday_day + " 00")]
    if len(midnight):
        df.loc[midnight[0], "holiday"] = "Test Day"
    clean, log = clean_traffic(pd.concat([df, extra, copy], ignore_index=True), cfg)
    removed = {r["rule"]: r["rows_removed"] for r in log}
    assert removed["drop_exact_duplicate_rows"] >= 1
    assert removed["drop_zero_kelvin_temperature"] >= 1 and removed["drop_impossible_rainfall"] >= 1
    assert removed["drop_closure_or_outage_hours"] >= 2
    assert (clean["traffic_volume"] >= cfg.cleaning.min_plausible_volume).all()
    assert clean["date_time"].is_unique and clean["date_time"].is_monotonic_increasing
    assert clean.loc[clean["date_time"] == pd.Timestamp(stamp), "weather_main"].iloc[0] == "Thunderstorm"
    if len(midnight):
        day_rows = clean[clean["date_time"].dt.strftime("%Y-%m-%d") == holiday_day]
        assert (day_rows["holiday"] == "Test Day").all()
        assert not day_rows["quiet_holiday"].any()  # a named day that is not on the quiet list is an ordinary day


def test_holidays_are_spread_named_and_only_quiet_ones_count(sample_raw, cfg):
    """The sample covers October 2012 to January 2013: Columbus Day, Veterans Day, Thanksgiving,
    Christmas and New Year's Day."""
    df = sample_raw.copy()
    # remove the midnight rows of Thanksgiving, the only rows that carry its name in the source
    df = df[~df["date_time"].str.startswith("2012-11-22 00")]
    assert not (df["holiday"] == "Thanksgiving Day").any()
    clean, log = clean_traffic(df, cfg)
    day = clean[clean["date_time"].dt.strftime("%Y-%m-%d") == "2012-11-22"]
    assert len(day) > 12 and (day["holiday"] == "Thanksgiving Day").all() and day["quiet_holiday"].all()
    named = {r["rule"]: r["values_changed"] for r in log}
    assert named["name_holidays_with_no_midnight_row"] >= len(day)
    columbus = clean[clean["date_time"].dt.strftime("%Y-%m-%d") == "2012-10-08"]
    assert (columbus["holiday"] == "Columbus Day").all() and not columbus["quiet_holiday"].any()
    assert parse_calendar(clean)["is_holiday"].sum() == clean["quiet_holiday"].sum() > 0
    evidence = holiday_evidence(clean, cfg.target).set_index("holiday")["share_of_normal"]
    assert evidence["Thanksgiving Day"] < 0.75 < evidence["Columbus Day"]


def test_split_is_chronological(sample_raw, cfg):
    clean, _ = clean_traffic(sample_raw, cfg)
    clean = clean.assign(date_time=clean["date_time"] + pd.DateOffset(years=4))  # move the sample into 2016-2017
    s = chronological_split(clean, cfg)
    t = clean["date_time"]
    if len(s["train"]) and len(s["validation"]):
        assert t.iloc[s["train"]].max() < t.iloc[s["validation"]].min()
    assert not set(s["train"]) & set(s["validation"]) and not set(s["validation"]) & set(s["test"])


def test_lags_follow_the_clock_not_the_row_order():
    stamps = pd.to_datetime(["2016-01-01 00:00", "2016-01-01 01:00", "2016-01-01 03:00", "2016-01-02 01:00"])
    clean = pd.DataFrame({"date_time": stamps, "traffic_volume": [10.0, 20.0, 30.0, 40.0]})
    lags = lag_frame(clean, "traffic_volume")
    assert lags["lag_1h"].tolist()[1] == 10.0
    assert np.isnan(lags["lag_1h"].tolist()[2])  # 02:00 is missing, so 03:00 has no previous hour
    assert lags["lag_24h"].tolist()[3] == 20.0


def test_metrics_and_block_bootstrap():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    m = regression_metrics(y, y + 1)
    assert m["rmse"] == pytest.approx(1.0) and m["mae"] == pytest.approx(1.0) and m["mse"] == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    yy = rng.normal(100, 10, 480)
    pp = yy + rng.normal(0, 5, 480)
    days = np.repeat(np.arange(20), 24)
    ci = block_bootstrap_ci(yy, pp, days, 200, seed=1)["rmse"]
    assert ci["low"] <= regression_metrics(yy, pp)["rmse"] <= ci["high"]


def test_paired_bootstrap_separates_a_better_model_from_an_equal_one():
    rng = np.random.default_rng(0)
    y = rng.normal(100, 10, 720)
    days = np.repeat(np.arange(30), 24)
    good, poor = y + rng.normal(0, 3, 720), y + rng.normal(0, 9, 720)
    better = paired_block_bootstrap(y, good, poor, days, 300, seed=1)
    assert better["rmse_reduction"]["low"] > 0 and better["share_of_resamples_model_better"] == 1.0
    assert better["rmse_reduction"]["low"] <= better["rmse_reduction"]["point"] <= better["rmse_reduction"]["high"]
    same = paired_block_bootstrap(y, good, good, days, 100, seed=1)
    assert same["rmse_difference"]["low"] == same["rmse_difference"]["high"] == 0.0


def test_slice_report_adds_up():
    stamps = pd.date_range("2018-01-01", periods=24 * 14, freq="h")  # two weeks from a Monday
    rows = pd.DataFrame({"date_time": stamps, "quiet_holiday": stamps.normalize() == pd.Timestamp("2018-01-01"),
                         "weather_main": np.where(stamps.day == 2, "Snow", "Clear")})
    y = np.full(len(rows), 1000.0)
    report = slice_report(rows, y, {"baseline": y + 100, "champion": y + 50})
    segments = {r["group"]: r for r in report["by_segment"]}
    assert sum(r["hours"] for r in segments.values()) == len(rows)
    assert segments["Quiet holiday"]["hours"] == 24 and segments["Weekend"]["hours"] == 96
    assert all(r["reduction"] == pytest.approx(0.5) for r in segments.values())
    assert sum(r["share_of_sq_error"] for r in segments.values()) == pytest.approx(1.0)
    assert {r["group"]: r["hours"] for r in report["by_weather"]} == {"Clear or cloudy": 312, "Snow": 24}
    assert report["by_period"][0]["group"] == "January to September" and len(report["by_period"]) == 1


def test_fit_verdict_follows_the_numbers():
    assert "no overfitting" in fit_verdict(390.0, 350.0, None)["text"]
    assert "training years are the harder ones" in fit_verdict(390.0, 350.0, None)["text"]
    assert "no sign of overfitting" in fit_verdict(350.0, 355.0, None)["text"]
    assert "mild overfitting" in fit_verdict(300.0, 350.0, None)["text"]
    assert "overfits" in fit_verdict(100.0, 350.0, None)["text"]
    curve = {"param": "depth", "values": [2, 8, 32], "train_rmse": [900, 300, 100], "val_rmse": [900, 350, 360], "chosen": 8}
    verdict = fit_verdict(300.0, 350.0, curve)
    assert verdict["best_capacity"] == 8 and "not underfitting" in verdict["text"]


def test_hour_of_week_baseline():
    X = pd.DataFrame({"weekday": [0, 0, 1, 1], "hour": [8, 8, 8, 9]})
    model = HourOfWeekBaseline().fit(X, [100, 200, 50, 70])
    assert model.predict(pd.DataFrame({"weekday": [0, 1, 6], "hour": [8, 9, 3]})).tolist() == [150.0, 70.0, 105.0]


def test_holiday_aware_baseline_treats_a_quiet_holiday_as_its_own_day():
    X = pd.DataFrame({"weekday": [0, 0, 0], "hour": [8, 8, 8], "is_holiday": [0, 0, 1]})
    model = HolidayAwareBaseline().fit(X, [100, 200, 30])
    new = pd.DataFrame({"weekday": [0, 0, 1], "hour": [8, 8, 8], "is_holiday": [0, 1, 1]})
    assert model.predict(new).tolist() == [150.0, 30.0, 30.0]  # a holiday on a Tuesday still uses the holiday row
    unseen = pd.DataFrame({"weekday": [0], "hour": [9], "is_holiday": [1]})
    assert model.predict(unseen).tolist() == [110.0]  # nothing known for that hour: overall mean


def test_psi():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert population_stability_index(a, rng.normal(0, 1, 5000)) < 0.05
    assert population_stability_index(a, rng.normal(1.5, 1, 5000)) > 0.25
