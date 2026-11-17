import numpy as np
import pandas as pd

from cryptomacro import config
from cryptomacro.analysis import (daily_event_comparison, event_matrix, event_window_profile,
                                  hourly_abs_returns, reaction_hours, release_times_utc)
from cryptomacro.clean import load_candles, load_cpi_release_dates
from tests.sample_data import write_sample_raw

WINTER = pd.DatetimeIndex(["2024-01-11"])  # release 13:30 UTC, so hour 0 is the 13:00 candle


def sample(tmp_path):
    write_sample_raw(tmp_path)
    return load_candles("BTC-USD", 3600, tmp_path), load_cpi_release_dates(tmp_path)


def random_walk_candles(start="2024-01-08", days=8):
    index = pd.date_range(start, periods=days * 24, freq="h", tz="UTC", name="time")
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(index))))
    return pd.DataFrame({"close": close, "volume": 1.0}, index=index)


def test_release_times_handle_daylight_saving():
    t = release_times_utc(pd.DatetimeIndex(["2024-01-11", "2024-07-11"]))
    assert t[0] == pd.Timestamp("2024-01-11 13:30", tz="UTC")
    assert t[1] == pd.Timestamp("2024-07-11 12:30", tz="UTC")


def test_event_matrix_hour_zero_is_candle_containing_release():
    index = pd.date_range("2024-01-08", periods=8 * 24, freq="h", tz="UTC")
    abs_ret = pd.Series(index.hour.astype(float), index=index)  # value = UTC hour number
    m = event_matrix(abs_ret, WINTER)
    assert list(m.columns) == list(range(-6, 13))
    assert m.loc["2024-01-11", 0] == 13 and m.loc["2024-01-11", -6] == 7 and m.loc["2024-01-11", 12] == 1
    outside = event_matrix(abs_ret, pd.DatetimeIndex(["2025-01-10"]))  # release after the data ends
    assert outside.isna().all(axis=None)


def test_event_matrix_keeps_the_callers_order_for_unsorted_dates():
    index = pd.date_range("2024-01-08", periods=8 * 24, freq="h", tz="UTC")
    abs_ret = pd.Series(np.arange(len(index), dtype=float), index=index)  # value = hours since start
    dates = pd.DatetimeIndex(["2024-01-12", "2023-06-01", "2024-01-10"])  # unsorted, middle one outside
    m = event_matrix(abs_ret, dates)
    assert list(m.index) == list(dates)
    sorted_rows = event_matrix(abs_ret, pd.DatetimeIndex(["2024-01-10", "2024-01-12"]))
    assert m.loc["2024-01-12"].equals(sorted_rows.loc["2024-01-12"])
    assert m.loc["2024-01-10"].equals(sorted_rows.loc["2024-01-10"])
    assert m.loc["2023-06-01"].isna().all()  # before the data: no candle within 59 minutes


def test_event_matrix_summer_hour_zero_is_12_utc():
    index = pd.date_range("2024-07-08", periods=8 * 24, freq="h", tz="UTC")
    abs_ret = pd.Series(index.hour.astype(float), index=index)
    assert event_matrix(abs_ret, pd.DatetimeIndex(["2024-07-11"])).loc["2024-07-11", 0] == 12


def test_hourly_abs_returns_value():
    candles = random_walk_candles()
    r = hourly_abs_returns(candles)
    t = candles.index[5]
    assert np.isclose(r[t], abs(np.log(candles["close"].iloc[5] / candles["close"].iloc[4])))
    assert np.isnan(r.iloc[0])


def test_event_matrix_missing_candle_is_nan_not_shifted():
    candles = random_walk_candles()
    full = event_matrix(hourly_abs_returns(candles), WINTER)
    gap = candles.drop(pd.Timestamp("2024-01-11 15:00", tz="UTC"))
    m = event_matrix(hourly_abs_returns(gap), WINTER)
    assert np.isnan(m.loc["2024-01-11", 2]) and np.isnan(m.loc["2024-01-11", 3])
    for hour in (1, 4):
        assert np.isfinite(m.loc["2024-01-11", hour])
        assert m.loc["2024-01-11", hour] == full.loc["2024-01-11", hour]


def test_profile_detects_three_hour_reaction(tmp_path):
    h, d = sample(tmp_path)
    prof = event_window_profile(h, d, draws=500)
    assert list(prof.index) == list(range(-6, 13))
    assert prof.loc[0, "cpi_mean"] > prof.loc[0, "base_high"]
    assert prof.loc[-3, "cpi_mean"] < prof.loc[-3, "base_high"]
    assert reaction_hours(prof) == 3


def test_profile_is_reproducible(tmp_path):
    h, d = sample(tmp_path)
    a = event_window_profile(h, d, draws=200, seed=1)
    b = event_window_profile(h, d, draws=200, seed=1)
    c = event_window_profile(h, d, draws=200, seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a["base_high"], c["base_high"])


def test_reaction_hours_none_when_no_reaction():
    hours = list(range(-6, 13))
    prof = pd.DataFrame({"cpi_mean": 1.0, "base_mean": 2.0, "base_low": 1.5, "base_high": 3.0}, index=hours)
    assert reaction_hours(prof) is None


def test_reaction_hours_counts_and_never_returns():
    hours = list(range(-6, 13))
    cpi = [1.0 if h < 0 else 5.0 if h < 2 else 1.0 for h in hours]
    prof = pd.DataFrame({"cpi_mean": cpi, "base_mean": 2.0, "base_low": 1.5, "base_high": 3.0}, index=hours)
    assert reaction_hours(prof) == 2
    prof["cpi_mean"] = 5.0
    assert reaction_hours(prof) == config.EVENT_HOURS_AFTER + 1


def test_daily_event_comparison_reads_minimum_returns_from_config(tmp_path, monkeypatch):
    h, d = sample(tmp_path)
    monkeypatch.setattr(config, "MIN_HOURLY_RETURNS_PER_DAY", 25)  # more than a day can have
    assert daily_event_comparison(h, d)["n_days"].tolist() == [0, 0]


def test_profile_band_reads_percentiles_from_config(tmp_path, monkeypatch):
    h, d = sample(tmp_path)
    narrow = event_window_profile(h, d, draws=200)
    monkeypatch.setattr(config, "BOOTSTRAP_BAND", (0.0, 100.0))  # min and max of the draws
    wide = event_window_profile(h, d, draws=200)
    assert (wide["base_high"] >= narrow["base_high"]).all() and (wide["base_low"] <= narrow["base_low"]).all()
    assert (wide["base_high"] > narrow["base_high"]).any()


def test_daily_event_comparison(tmp_path):
    h, d = sample(tmp_path)
    c = daily_event_comparison(h, d)
    assert list(c.index) == [False, True] and c.index.name == "cpi_day"
    assert list(c.columns) == ["realized_vol_mean", "realized_vol_std", "volume_mean", "n_days"]
    assert c.loc[True, "realized_vol_mean"] > c.loc[False, "realized_vol_mean"]
    assert c.loc[True, "volume_mean"] > c.loc[False, "volume_mean"]
    assert c.loc[True, "n_days"] == len(d)
