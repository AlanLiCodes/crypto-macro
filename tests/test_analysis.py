import math

import numpy as np
import pandas as pd
import pytest

from cryptomacro import config
from cryptomacro.analysis import (annualized_volatility, calendar_day_log_returns, correlation_peak,
                                  describe_changes, hedge_correlation_by_year, hedge_correlation_matrix,
                                  rate_beta_by_regime, rolling_correlation, summary_table,
                                  weekly_comparison, zscore)
from cryptomacro.clean import build_daily_panel, load_candles
from tests.sample_data import BOND_ONLY_HOLIDAYS, write_sample_raw


@pytest.fixture
def sample_raw(tmp_path):
    write_sample_raw(tmp_path)
    return tmp_path


@pytest.fixture
def sample_panel(sample_raw):
    return build_daily_panel(sample_raw)


@pytest.fixture
def btc_daily_close(sample_raw):
    return load_candles("BTC-USD", 86400, sample_raw)["close"]


def perfect_panel(rows):
    """A panel where BTC_ret is exactly twice SP500_ret."""
    rng = np.random.default_rng(0)
    sp = rng.normal(0, 0.01, rows)
    index = pd.bdate_range("2022-01-03", periods=rows, name="date")
    return pd.DataFrame({"BTC_ret": 2 * sp, "SP500_ret": sp}, index=index)


def test_rolling_correlation_perfect():
    c = rolling_correlation(perfect_panel(200), "BTC", "SP500", 90)
    assert c.name == "BTC vs SP500"
    needed = math.ceil(config.MIN_WINDOW_FRACTION * 90)  # 72 rows
    assert c.iloc[:needed - 1].isna().all()
    assert c.iloc[needed - 1:].to_numpy() == pytest.approx(1.0)


def test_rolling_correlation_skips_missing_days_instead_of_using_zero():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2022-01-03", periods=120, name="date")
    panel = pd.DataFrame({"BTC_ret": rng.normal(0, 0.02, 120), "SP500_ret": rng.normal(0, 0.01, 120)},
                         index=index)
    panel.iloc[[100, 105], panel.columns.get_loc("SP500_ret")] = np.nan
    c = rolling_correlation(panel, "BTC", "SP500", 90)
    window = panel.iloc[110 - 89:111].dropna()  # the 90 rows ending at row 110, complete pairs only
    assert len(window) == 88
    assert c.iloc[110] == pytest.approx(window["BTC_ret"].corr(window["SP500_ret"]))


def test_bond_market_holidays_do_not_blank_the_rolling_correlation(sample_panel):
    # The stock market is open on these days and the bond market is closed.
    for day in BOND_ONLY_HOLIDAYS:
        assert np.isnan(sample_panel.loc[day, "DGS10"]) and not np.isnan(sample_panel.loc[day, "SP500"])
    c = rolling_correlation(sample_panel, "BTC", "DGS10", 90)
    assert c.iloc[90:].notna().all()  # no NaN after the warm-up


def test_correlation_peak_finds_max_and_date():
    index = pd.bdate_range("2022-01-03", periods=5)
    corr = pd.Series([np.nan, 0.2, 0.9, 0.5, -0.3], index=index)
    date, value = correlation_peak(corr)
    assert date == index[2] and value == pytest.approx(0.9)


def test_window_longer_than_data_gives_nan_not_error():
    # 50 rows are fewer than the 72 needed for a 90-day window.
    c = rolling_correlation(perfect_panel(200).iloc[:50], "BTC", "SP500", 90)
    assert c.isna().all()
    date, value = correlation_peak(c)
    assert date is None and np.isnan(value)


def test_annualized_volatility_scaling():
    r = np.array([0.01, -0.01, 0.02, -0.02])
    assert annualized_volatility(r, 365) == pytest.approx(np.std(r, ddof=1) * np.sqrt(365))


def test_annualized_volatility_ignores_nan():
    r = np.array([np.nan, 0.01, -0.01, 0.02, -0.02])
    assert annualized_volatility(r, 252) == pytest.approx(np.std(r[1:], ddof=1) * np.sqrt(252))


def test_weekly_comparison_shape(sample_panel):
    t = weekly_comparison(sample_panel, "BTC", "SP500")
    assert list(t.index) == ["daily", "weekly"]
    assert list(t.columns) == ["correlation", "n"]
    assert t.loc["weekly", "n"] < t.loc["daily", "n"]
    assert t["correlation"].between(-1, 1).all()


def test_weekly_comparison_rate_series_uses_differences(sample_panel):
    # DGS10 is a difference series; the weekly row must still be computed without error.
    t = weekly_comparison(sample_panel, "ETH", "DGS10")
    assert list(t.index) == ["daily", "weekly"] and t["n"].gt(0).all()


def test_calendar_day_log_returns_never_span_a_missing_day():
    days = pd.date_range("2022-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 110.0, 121.0, 133.1, 146.41], index=days).drop(days[2])
    r = calendar_day_log_returns(close)
    assert len(r) == 5  # one row per calendar day, the missing one included
    assert r.iloc[1] == pytest.approx(np.log(1.1))
    assert np.isnan(r.iloc[0]) and np.isnan(r.iloc[2]) and np.isnan(r.iloc[3])
    assert r.iloc[4] == pytest.approx(np.log(1.1))


def test_summary_table_coin_volatility_uses_every_calendar_day(sample_panel, btc_daily_close):
    t = summary_table(sample_panel, "BTC", "SP500", 90, coin_daily_close=btc_daily_close)
    expected = np.std(np.diff(np.log(btc_daily_close.to_numpy())), ddof=1) * np.sqrt(365)
    assert t.loc["Annualized volatility (coin, x sqrt 365)", "value"] == pytest.approx(expected)
    # Trading-day returns (weekends folded into Monday) scaled by sqrt(365) would be a different number.
    trading_day_version = annualized_volatility(sample_panel["BTC_ret"], 365)
    assert t.loc["Annualized volatility (coin, x sqrt 365)", "value"] != pytest.approx(trading_day_version)
    assert t.loc["Annualized volatility (S&P 500, x sqrt 252)", "value"] == pytest.approx(
        annualized_volatility(sample_panel["SP500_ret"], 252))


def test_summary_table_on_sample(sample_panel, btc_daily_close):
    t = summary_table(sample_panel, "BTC", "SP500", 90, coin_daily_close=btc_daily_close)
    assert list(t.index) == [
        "Days in sample", "Mean daily return (coin)", "Annualized volatility (coin, x sqrt 365)",
        "Annualized volatility (S&P 500, x sqrt 252)", "Full-sample correlation",
        "Peak rolling correlation", "Peak date", "Latest rolling correlation"]
    assert t.loc["Full-sample correlation", "value"] > 0.3
    assert t.loc["Days in sample", "value"] == len(sample_panel)
    assert isinstance(t.loc["Peak date", "value"], pd.Timestamp)
    assert t.loc["Peak rolling correlation", "value"] >= t.loc["Latest rolling correlation", "value"]


def test_summary_table_short_window_has_nan_and_no_peak_date(sample_panel, btc_daily_close):
    t = summary_table(sample_panel.iloc[:50], "BTC", "SP500", 90, coin_daily_close=btc_daily_close)
    assert t.loc["Peak date", "value"] is None
    assert np.isnan(t.loc["Latest rolling correlation", "value"])


def regime_panel(sizes, nan_regime_rows=0):
    """BTC_ret = 0.001 - 0.5 * DGS10_chg when raising, 0.002 + 0.25 * DGS10_chg when cutting.

    sizes maps regime name to a row count; holding rows get random values. Extra rows with a NaN regime
    follow the others and use the "raising" relationship.
    """
    rng = np.random.default_rng(1)
    regimes = [name for name, count in sizes.items() for _ in range(count)] + [None] * nan_regime_rows
    rows = len(regimes)
    dgs = rng.normal(0, 0.05, rows)
    btc = np.where(np.array(regimes, dtype=object) == "cutting", 0.002 + 0.25 * dgs,
                   np.where(np.array(regimes, dtype=object) == "holding", rng.normal(0, 0.01, rows),
                            0.001 - 0.5 * dgs))
    index = pd.bdate_range("2022-01-03", periods=rows, name="date")
    regime = pd.Categorical(regimes, categories=config.REGIME_LABELS)
    return pd.DataFrame({"BTC_ret": btc, "DGS10_chg": dgs, "regime": regime}, index=index)


def test_rate_beta_recovers_known_slope():
    b = rate_beta_by_regime(regime_panel({"cutting": 100, "holding": 100, "raising": 100}), "BTC")
    assert list(b.index) == ["cutting", "holding", "raising", "all"]
    assert list(b.columns) == ["alpha", "beta", "corr", "n"]
    assert b.loc["raising", "beta"] == pytest.approx(-0.5)
    assert b.loc["raising", "alpha"] == pytest.approx(0.001)
    assert b.loc["raising", "corr"] == pytest.approx(-1.0)
    assert b.loc["cutting", "beta"] == pytest.approx(0.25)
    assert b.loc["cutting", "alpha"] == pytest.approx(0.002)
    assert b.loc["raising", "n"] == 100
    assert b.loc["all", "n"] == 300


def test_rate_beta_rows_follow_the_shared_regime_labels():
    assert config.REGIME_LABELS == ["cutting", "holding", "raising"]
    b = rate_beta_by_regime(regime_panel({"cutting": 10, "holding": 10, "raising": 10}), "BTC")
    assert list(b.index) == config.REGIME_LABELS + ["all"]


def test_rate_beta_small_group_is_nan():
    b = rate_beta_by_regime(regime_panel({"cutting": 50, "holding": 2, "raising": 50}), "BTC")
    assert b.loc["holding", "n"] == 2
    assert b.loc["holding", ["alpha", "beta", "corr"]].isna().all()
    assert b.loc["raising", "beta"] == pytest.approx(-0.5)


def test_rate_beta_missing_regime_still_has_a_row():
    b = rate_beta_by_regime(regime_panel({"cutting": 50, "raising": 50}), "BTC")
    assert list(b.index) == ["cutting", "holding", "raising", "all"]
    assert b.loc["holding", "n"] == 0
    assert b.loc["holding", ["alpha", "beta", "corr"]].isna().all()


def test_rate_beta_ignores_nan_rows():
    p = regime_panel({"cutting": 50, "holding": 50, "raising": 50})
    p.iloc[120, p.columns.get_loc("DGS10_chg")] = np.nan  # a "raising" row
    b = rate_beta_by_regime(p, "BTC")
    assert b.loc["raising", "n"] == 49
    assert b.loc["all", "n"] == 149
    assert b.loc["raising", "beta"] == pytest.approx(-0.5)


def test_rate_beta_all_includes_rows_without_regime():
    b = rate_beta_by_regime(regime_panel({"cutting": 20, "holding": 20, "raising": 20}, nan_regime_rows=10), "BTC")
    assert b.loc["all", "n"] == 70
    assert b.loc[["cutting", "holding", "raising"], "n"].sum() == 60


def test_rate_beta_zero_variance_is_nan_without_warning():
    p = regime_panel({"cutting": 20, "holding": 20, "raising": 20})
    p.loc[p["regime"] == "holding", "DGS10_chg"] = 0.0
    b = rate_beta_by_regime(p, "BTC")
    assert b.loc["holding", "n"] == 20
    assert b.loc["holding", ["alpha", "beta", "corr"]].isna().all()
    assert b.loc["raising", "beta"] == pytest.approx(-0.5)


def test_zscore():
    assert zscore(pd.Series([1., 2., 3.])).tolist() == pytest.approx([-1, 0, 1])


def hedge_panel(rows=300):
    """BTC moves with breakeven inflation and against the dollar index."""
    rng = np.random.default_rng(2)
    inflation = rng.normal(0, 0.02, rows)
    dollar = rng.normal(0, 0.003, rows)
    btc = inflation / 0.02 * 0.01 - dollar / 0.003 * 0.01 + rng.normal(0, 0.003, rows)
    index = pd.bdate_range("2022-01-03", periods=rows, name="date")
    return pd.DataFrame({"BTC_ret": btc, "T10YIE_chg": inflation, "DTWEXBGS_ret": dollar}, index=index)


def test_hedge_matrix_signs():
    m = hedge_correlation_matrix(hedge_panel())
    labels = ["BTC", "Breakeven inflation", "Dollar index"]
    assert list(m.index) == labels and list(m.columns) == labels
    assert m.loc["BTC", "Breakeven inflation"] > 0.5
    assert m.loc["BTC", "Dollar index"] < -0.5
    assert np.allclose(np.diag(m), 1.0)
    assert np.allclose(m, m.T)


def test_hedge_matrix_uses_coin_label():
    p = hedge_panel().rename(columns={"BTC_ret": "ETH_ret"})
    assert list(hedge_correlation_matrix(p, "ETH").index)[0] == "ETH"


def test_hedge_matrix_too_few_rows_is_all_nan():
    m = hedge_correlation_matrix(hedge_panel().iloc[:2])
    assert m.shape == (3, 3)
    assert m.isna().all().all()


def test_hedge_matrix_drops_incomplete_rows():
    p = hedge_panel()
    p.iloc[:297, p.columns.get_loc("T10YIE_chg")] = np.nan  # only 3 complete rows remain
    m = hedge_correlation_matrix(p)
    assert not m.isna().any().any()


def test_hedge_by_year_index(sample_panel):
    t = hedge_correlation_by_year(sample_panel)
    assert list(t.index) == [2022, 2023]
    assert t.index.name == "year"
    assert list(t.columns) == ["corr_inflation", "corr_dollar", "n"]
    assert (t["n"] > 3).all()


def test_hedge_by_year_values_and_small_year():
    p = hedge_panel(300)  # starts 2022-01-03, runs into 2023
    extra = p.iloc[-2:].copy()
    extra.index = pd.to_datetime(["2024-01-02", "2024-01-03"])
    t = hedge_correlation_by_year(pd.concat([p, extra]))
    assert list(t.index) == [2022, 2023, 2024]
    assert t.loc[2022, "corr_inflation"] > 0.5 and t.loc[2022, "corr_dollar"] < -0.5
    assert t.loc[2024, "n"] == 2
    assert np.isnan(t.loc[2024, "corr_inflation"]) and np.isnan(t.loc[2024, "corr_dollar"])


def test_describe_changes_covers_the_eight_change_columns(sample_panel):
    d = describe_changes(sample_panel)
    assert list(d.columns) == ["BTC_ret", "ETH_ret", "SP500_ret", "NASDAQCOM_ret",
                               "DGS10_chg", "DFF_chg", "T10YIE_chg", "DTWEXBGS_ret"]
    assert list(d.index) == ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    assert d.loc["count", "DGS10_chg"] == sample_panel["DGS10_chg"].notna().sum()
    assert d.loc["mean", "BTC_ret"] == pytest.approx(sample_panel["BTC_ret"].mean())


def test_tunables_live_in_config():
    assert config.MIN_WINDOW_FRACTION == 0.8
    assert config.MIN_HOURLY_RETURNS_PER_DAY == 12
    assert config.CRYPTO_DAYS_PER_YEAR == 365 and config.STOCK_DAYS_PER_YEAR == 252
    assert config.BOOTSTRAP_BAND == (2.5, 97.5)
    assert config.WEEKLY_RULE == "W-FRI"


def test_weekly_comparison_reads_the_week_rule_from_config(sample_panel, monkeypatch):
    friday_weeks = weekly_comparison(sample_panel, "BTC", "SP500")
    monkeypatch.setattr(config, "WEEKLY_RULE", "W-WED")
    wednesday_weeks = weekly_comparison(sample_panel, "BTC", "SP500")
    assert wednesday_weeks.loc["weekly", "correlation"] != friday_weeks.loc["weekly", "correlation"]
