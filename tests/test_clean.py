import json

import numpy as np
import pandas as pd
import pytest

from cryptomacro import config
from cryptomacro.clean import (DataMissingError, build_daily_panel, label_regimes, load_candles,
                               load_cpi_release_dates, load_dropped_cpi_dates, load_fred_series,
                               log_returns, split_cpi_release_dates)
from tests.sample_data import BOND_ONLY_HOLIDAYS, write_sample_raw


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def ts(day):
    return int(pd.Timestamp(day, tz="UTC").timestamp())


def test_load_candles_sorts_dedupes_and_types(tmp_path):
    # Rows are [time, low, high, open, close, volume], newest first, as the API returns them.
    # 2020-01-02 appears in both files.
    write(tmp_path / "coinbase" / "BTC-USD_86400_20200101T0000.json",
          [[ts("2020-01-02"), 6900, 7300, 7000, 7100, 11], [ts("2020-01-01"), 6800, 7250, 7195, 7200, 10]])
    write(tmp_path / "coinbase" / "BTC-USD_86400_20200102T0000.json",
          [[ts("2020-01-03"), 7000, 7400, 7100, 7350, 12], [ts("2020-01-02"), 6900, 7300, 7000, 7100, 11]])
    df = load_candles("BTC-USD", 86400, tmp_path)
    assert list(df.columns) == ["low", "high", "open", "close", "volume"]
    assert df.index.name == "time"
    assert df.index.is_monotonic_increasing and df.index.is_unique and str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2020-01-01", tz="UTC") and df["close"].iloc[0] == 7200.0
    assert len(df) == 3
    assert (df.dtypes == float).all()


def test_load_candles_ignores_other_products_and_granularities(tmp_path):
    write(tmp_path / "coinbase" / "BTC-USD_86400_20200101T0000.json", [[ts("2020-01-01"), 1, 2, 3, 4, 5]])
    write(tmp_path / "coinbase" / "BTC-USD_3600_20200101T0000.json", [[ts("2020-01-05"), 1, 2, 3, 4, 5]])
    write(tmp_path / "coinbase" / "ETH-USD_86400_20200101T0000.json", [[ts("2020-01-06"), 1, 2, 3, 4, 5]])
    assert list(load_candles("BTC-USD", 86400, tmp_path).index) == [pd.Timestamp("2020-01-01", tz="UTC")]


def test_load_candles_tolerates_empty_window(tmp_path):
    write(tmp_path / "coinbase" / "BTC-USD_86400_20200101T0000.json", [])
    write(tmp_path / "coinbase" / "BTC-USD_86400_20200102T0000.json", [[ts("2020-01-02"), 1, 2, 3, 4, 5]])
    df = load_candles("BTC-USD", 86400, tmp_path)
    assert len(df) == 1 and df["close"].iloc[0] == 4.0


def test_load_candles_missing_raises_with_command(tmp_path):
    with pytest.raises(DataMissingError, match="python -m cryptomacro.acquire") as excinfo:
        load_candles("BTC-USD", 86400, tmp_path)
    assert "BTC-USD_86400_*.json" in str(excinfo.value)  # says which files are missing


def test_load_fred_series_dot_becomes_nan(tmp_path):
    write(tmp_path / "fred" / "SP500.json", {"observations": [
        {"date": "2020-01-02", "value": "3257.85"},
        {"date": "2020-01-03", "value": "."},
        {"date": "2020-01-06", "value": "3234.85"}]})
    s = load_fred_series("SP500", tmp_path)
    assert np.isnan(s.iloc[1]) and s.iloc[0] == 3257.85 and s.name == "SP500"
    assert s.index.name == "date" and s.index.tz is None
    assert s.index[2] == pd.Timestamp("2020-01-06") and s.dtype == float


def test_load_fred_series_missing_raises_with_command(tmp_path):
    with pytest.raises(DataMissingError, match="python -m cryptomacro.acquire"):
        load_fred_series("SP500", tmp_path)


def test_load_cpi_release_dates_limited_to_period(tmp_path):
    write(tmp_path / "fred" / "cpi_release_dates.json", {"release_dates": [
        {"release_id": 10, "date": "2026-10-14"},
        {"release_id": 10, "date": "2020-01-14"},
        {"release_id": 10, "date": "2019-12-11"}]})
    dates = load_cpi_release_dates(tmp_path)
    assert list(dates) == [pd.Timestamp("2020-01-14")]
    assert dates.tz is None


def test_load_cpi_release_dates_sorted_and_unique(tmp_path):
    write(tmp_path / "fred" / "cpi_release_dates.json", {"release_dates": [
        {"release_id": 10, "date": "2020-03-11"}, {"release_id": 10, "date": "2020-02-13"},
        {"release_id": 10, "date": "2020-03-11"}]})
    assert list(load_cpi_release_dates(tmp_path)) == [pd.Timestamp("2020-02-13"), pd.Timestamp("2020-03-11")]


def test_split_cpi_release_dates_keeps_the_latest_date_of_each_month():
    dates = pd.DatetimeIndex(["2020-03-11", "2020-02-13", "2020-03-02", "2020-04-10", "2020-03-20"])
    kept, dropped = split_cpi_release_dates(dates)
    assert list(kept) == [pd.Timestamp(d) for d in ["2020-02-13", "2020-03-20", "2020-04-10"]]
    assert list(dropped) == [pd.Timestamp(d) for d in ["2020-03-02", "2020-03-11"]]


def test_split_cpi_release_dates_with_one_date_per_month_drops_nothing():
    kept, dropped = split_cpi_release_dates(pd.DatetimeIndex(["2020-01-14", "2020-02-13"]))
    assert len(kept) == 2 and len(dropped) == 0


def test_cpi_loaders_return_kept_and_dropped_dates(tmp_path):
    write(tmp_path / "fred" / "cpi_release_dates.json", {"release_dates": [
        {"release_id": 10, "date": "2020-01-14"}, {"release_id": 10, "date": "2020-02-03"},
        {"release_id": 10, "date": "2020-02-13"}, {"release_id": 10, "date": "2019-12-02"},
        {"release_id": 10, "date": "2019-12-11"}]})  # December 2019 is outside the period
    assert list(load_cpi_release_dates(tmp_path)) == [pd.Timestamp("2020-01-14"), pd.Timestamp("2020-02-13")]
    assert list(load_dropped_cpi_dates(tmp_path)) == [pd.Timestamp("2020-02-03")]


def test_load_cpi_release_dates_missing_raises(tmp_path):
    with pytest.raises(DataMissingError):
        load_cpi_release_dates(tmp_path)


def test_sample_data_loads(tmp_path):
    write_sample_raw(tmp_path)
    assert len(load_candles("ETH-USD", 3600, tmp_path)) > 10000


# ---------- panel, returns, regimes ----------

def test_log_returns_matches_formula():
    np.testing.assert_allclose(log_returns(np.array([100., 110., 99.])), [np.log(1.1), np.log(0.9)])


def test_label_regimes_thresholds():
    # 1.00 for 100 days, then 1.125 for 100 days, then 0.875 for 100 days
    values = [1.0] * 100 + [1.125] * 100 + [0.875] * 100
    dff = pd.Series(values, index=pd.date_range("2022-01-01", periods=300, freq="D"))
    r = label_regimes(dff)
    assert r.iloc[95] == "holding" and r.iloc[150] == "raising" and r.iloc[250] == "cutting"
    assert r.iloc[:90].isna().all()
    assert list(r.cat.categories) == config.REGIME_LABELS


def test_label_regimes_boundaries_are_exact_and_gaps_are_filled():
    # Day 100 onward is exactly 0.125 above, day 200 onward exactly 0.125 below the level 90 days before.
    values = [1.0] * 100 + [1.125] * 100 + [1.0] * 100
    dff = pd.Series(values, index=pd.date_range("2022-01-01", periods=300, freq="D"))
    r = label_regimes(dff.drop(dff.index[[120, 121, 122]]))  # missing days must not break the lookback
    assert r.iloc[150] == "raising"
    assert r.iloc[250] == "cutting"  # 1.0 - 1.125 = -0.125 exactly
    assert len(r) == 300


def test_panel_only_trading_days_and_weekend_folds_into_monday(tmp_path):
    write_sample_raw(tmp_path)
    p = build_daily_panel(tmp_path)
    assert p.index.name == "date" and p.index.tz is None
    assert (p.index.day_of_week < 5).all() and p["SP500"].notna().all()
    assert pd.Timestamp("2022-07-04") not in p.index  # holiday
    fri = p.index[p.index.day_of_week == 4][1]
    mon = p.index[p.index.get_loc(fri) + 1]
    assert mon - fri == pd.Timedelta(days=3)
    assert p.loc[mon, "BTC_ret"] == pytest.approx(np.log(p.loc[mon, "BTC"] / p.loc[fri, "BTC"]))
    assert p.loc[mon, "SP500_ret"] == pytest.approx(np.log(p.loc[mon, "SP500"] / p.loc[fri, "SP500"]))
    assert p.loc[mon, "DGS10_chg"] == pytest.approx(p.loc[mon, "DGS10"] - p.loc[fri, "DGS10"])
    assert p.loc[mon, "DTWEXBGS_ret"] == pytest.approx(p.loc[mon, "DTWEXBGS"] / p.loc[fri, "DTWEXBGS"] - 1)


def test_panel_has_expected_columns(tmp_path):
    write_sample_raw(tmp_path)
    p = build_daily_panel(tmp_path)
    assert set(p.columns) == {"BTC", "ETH", "SP500", "NASDAQCOM", "DGS10", "DFF", "T10YIE", "DTWEXBGS",
                              "BTC_ret", "ETH_ret", "SP500_ret", "NASDAQCOM_ret",
                              "DGS10_chg", "DFF_chg", "T10YIE_chg", "DTWEXBGS_ret", "regime"}
    assert set(p["regime"].dropna().unique()) == {"cutting", "holding", "raising"}


def test_panel_first_days_have_a_regime_and_stay_inside_the_sample(tmp_path):
    # DFF is saved from 100 days before the first trading day, so the 90-day lookback is covered.
    write_sample_raw(tmp_path)
    p = build_daily_panel(tmp_path)
    assert load_fred_series("DFF", tmp_path).index.min() < p.index.min() - pd.Timedelta(days=90)
    assert p["regime"].notna().all()
    assert p.index.min() == pd.Timestamp("2022-01-03")  # the earlier DFF days do not add panel rows


def test_panel_no_inf_and_nan_propagates(tmp_path):
    write_sample_raw(tmp_path)
    path = tmp_path / "fred" / "DGS10.json"
    data = json.loads(path.read_text())
    # Blank one ordinary mid-week trading day (the holidays are 2022-07-04 and 2023-11-23).
    day = pd.Timestamp("2022-09-14")
    for obs in data["observations"]:
        if obs["date"] == f"{day:%Y-%m-%d}":
            obs["value"] = "."
    path.write_text(json.dumps(data))

    p = build_daily_panel(tmp_path)
    next_day = p.index[p.index.get_loc(day) + 1]
    assert np.isnan(p.loc[day, "DGS10"])  # not forward-filled
    assert not np.isinf(p.select_dtypes("number")).any().any()
    assert np.isnan(p.loc[day, "DGS10_chg"]) and np.isnan(p.loc[next_day, "DGS10_chg"])
    # First row, the blank day and the day after, plus two per bond-market-only holiday.
    assert p["DGS10_chg"].isna().sum() == 3 + 2 * len(BOND_ONLY_HOLIDAYS)
