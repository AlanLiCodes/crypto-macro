"""Statistics behind the dashboard: rolling correlation, volatility, weekly comparison, summary table.

This module only works on the panel built by clean.py. It does not touch files or the network.
"""
import math

import numpy as np
import pandas as pd

from cryptomacro import config


def rolling_correlation(panel: pd.DataFrame, coin: str, series: str, window: int = config.DEFAULT_WINDOW) -> pd.Series:
    """Correlation of the coin's daily return with the chosen series' daily change over a rolling window.

    A window is used when at least MIN_WINDOW_FRACTION of its days (72 of 90) have both values.
    pandas leaves out the days where either value is missing (a bond-market holiday, for example),
    so a missing day is skipped, never treated as zero. Windows with fewer complete days give NaN.
    """
    coin_returns = panel[f"{coin}_ret"]
    changes = panel[config.CHANGE_COLUMN[series]]
    min_periods = math.ceil(config.MIN_WINDOW_FRACTION * window)
    corr = coin_returns.rolling(window, min_periods=min_periods).corr(changes)
    return corr.rename(f"{coin} vs {series}")


def correlation_peak(corr: pd.Series) -> tuple[pd.Timestamp | None, float]:
    """The date and value of the highest correlation. (None, nan) if every value is NaN."""
    if corr.isna().all():
        return None, float("nan")
    date = corr.idxmax()  # idxmax skips NaN
    return date, float(corr.loc[date])


def annualized_volatility(returns, periods_per_year: int) -> float:
    """Sample standard deviation of the returns (NaN ignored) times the square root of periods per year."""
    return float(np.nanstd(np.asarray(returns, dtype=float), ddof=1) * np.sqrt(periods_per_year))


def calendar_day_log_returns(daily_close: pd.Series) -> pd.Series:
    """Log return from each calendar day's close to the next day's close (crypto trades every day).

    The index is made complete first, so a missing daily candle gives NaN instead of a two-day return.
    """
    every_day = pd.date_range(daily_close.index.min(), daily_close.index.max(), freq="D")
    return np.log(daily_close.reindex(every_day)).diff()


def describe_changes(panel: pd.DataFrame) -> pd.DataFrame:
    """pandas describe() of the eight daily change columns: the two coins and the six FRED series."""
    columns = [f"{coin}_ret" for coin in config.PRODUCTS] + list(config.CHANGE_COLUMN.values())
    return panel[columns].describe()


def _correlation_and_count(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    """Correlation over the rows where both values exist, and how many such rows there are."""
    both = pd.concat([x, y], axis=1).dropna()
    return float(both.iloc[:, 0].corr(both.iloc[:, 1])), len(both)


def _weekly_change(levels: pd.DataFrame, column: str, change_column: str) -> pd.Series:
    """Change of one column between consecutive weekly levels, defined like the daily change column.

    The daily panel uses a difference for *_chg columns, a percent change for DTWEXBGS_ret and a
    log return for every other *_ret column (prices and the coins).
    """
    if change_column.endswith("_chg"):
        return levels[column].diff()
    if change_column == "DTWEXBGS_ret":
        return levels[column].pct_change(fill_method=None)
    return np.log(levels[column]).diff()


def weekly_comparison(panel: pd.DataFrame, coin: str, series: str) -> pd.DataFrame:
    """Correlation of the coin with the series using daily changes and using weekly changes.

    Weekly levels are the last value of each week (weeks end on Friday); weekly changes are then
    computed from those levels. Row "daily" uses the panel's own change columns.
    """
    change_column = config.CHANGE_COLUMN[series]
    daily_corr, daily_n = _correlation_and_count(panel[f"{coin}_ret"], panel[change_column])

    weekly_levels = panel[[coin, series]].resample(config.WEEKLY_RULE).last()
    weekly_coin = np.log(weekly_levels[coin]).diff()  # coins are always log returns
    weekly_series = _weekly_change(weekly_levels, series, change_column)
    weekly_corr, weekly_n = _correlation_and_count(weekly_coin, weekly_series)

    return pd.DataFrame({"correlation": [daily_corr, weekly_corr], "n": [daily_n, weekly_n]},
                        index=["daily", "weekly"])


def summary_table(panel: pd.DataFrame, coin: str, series: str, window: int = config.DEFAULT_WINDOW, *,
                  coin_daily_close: pd.Series) -> pd.DataFrame:
    """One row per statistic, column "value" (mixed types: numbers and a date).

    coin_daily_close is the coin's daily candle close for every calendar day. The coin's volatility
    uses those 7-days-a-week returns (x sqrt 365); the S&P 500 uses the panel's trading-day
    returns (x sqrt 252). Everything else comes from the trading-day panel.
    """
    coin_returns = panel[f"{coin}_ret"]
    corr = rolling_correlation(panel, coin, series, window)
    peak_date, peak_value = correlation_peak(corr)
    latest = corr.dropna()
    full_corr, _ = _correlation_and_count(coin_returns, panel[config.CHANGE_COLUMN[series]])
    coin_volatility = annualized_volatility(calendar_day_log_returns(coin_daily_close),
                                            config.CRYPTO_DAYS_PER_YEAR)

    values = {
        "Days in sample": len(panel),
        "Mean daily return (coin)": float(coin_returns.mean()),
        "Annualized volatility (coin, x sqrt 365)": coin_volatility,
        "Annualized volatility (S&P 500, x sqrt 252)": annualized_volatility(panel["SP500_ret"],
                                                                             config.STOCK_DAYS_PER_YEAR),
        "Full-sample correlation": full_corr,
        "Peak rolling correlation": peak_value,
        "Peak date": peak_date,
        "Latest rolling correlation": float(latest.iloc[-1]) if len(latest) else float("nan"),
    }
    return pd.DataFrame({"value": pd.Series(values, dtype=object)})


def _rate_fit(coin_returns: pd.Series, rate_changes: pd.Series) -> dict:
    """Least-squares line coin = alpha + beta * rate change, over the rows where both values exist."""
    both = pd.concat([coin_returns, rate_changes], axis=1).dropna()
    n = len(both)
    result = {"alpha": np.nan, "beta": np.nan, "corr": np.nan, "n": n}
    # Too few rows, or a rate change that never moves, gives no meaningful slope.
    if n < config.MIN_ROWS_FOR_STATS or both.iloc[:, 1].std() == 0:
        return result
    y = both.iloc[:, 0].to_numpy()
    x = both.iloc[:, 1].to_numpy()
    design = np.column_stack([np.ones(n), x])  # first column fits the intercept
    (alpha, beta), *_ = np.linalg.lstsq(design, y, rcond=None)
    result.update(alpha=float(alpha), beta=float(beta), corr=float(np.corrcoef(x, y)[0, 1]))
    return result


def rate_beta_by_regime(panel: pd.DataFrame, coin: str) -> pd.DataFrame:
    """Sensitivity of the coin's daily return to the daily change in the 10-year yield, per Fed regime.

    Rows are "cutting", "holding", "raising" (always all three, n = 0 if a regime never occurs) and
    "all". "all" uses every row with both values, including rows whose regime is NaN.
    """
    coin_returns = panel[f"{coin}_ret"]
    rate_changes = panel["DGS10_chg"]
    rows = {}
    for regime in config.REGIME_LABELS:
        in_regime = (panel["regime"] == regime).to_numpy()
        rows[regime] = _rate_fit(coin_returns[in_regime], rate_changes[in_regime])
    rows["all"] = _rate_fit(coin_returns, rate_changes)
    return pd.DataFrame.from_dict(rows, orient="index")[["alpha", "beta", "corr", "n"]]


def zscore(s: pd.Series) -> pd.Series:
    """Standardize: subtract the mean, divide by the sample standard deviation."""
    return (s - s.mean()) / s.std(ddof=1)


def _hedge_columns(panel: pd.DataFrame, coin: str) -> pd.DataFrame:
    """The coin's return, breakeven inflation change and dollar index return, complete rows only."""
    columns = [f"{coin}_ret", "T10YIE_chg", "DTWEXBGS_ret"]
    return panel[columns].dropna()


def hedge_correlation_matrix(panel: pd.DataFrame, coin: str = "BTC") -> pd.DataFrame:
    """3x3 correlation of the coin with breakeven inflation and the dollar index (complete rows only)."""
    labels = [coin, "Breakeven inflation", "Dollar index"]
    data = _hedge_columns(panel, coin)
    if len(data) < config.MIN_ROWS_FOR_STATS:
        return pd.DataFrame(np.nan, index=labels, columns=labels)
    return pd.DataFrame(np.corrcoef(data.to_numpy(), rowvar=False), index=labels, columns=labels)


def hedge_correlation_by_year(panel: pd.DataFrame, coin: str = "BTC") -> pd.DataFrame:
    """Per calendar year: coin correlation with breakeven inflation and with the dollar index, and n."""
    data = _hedge_columns(panel, coin)
    rows = {}
    for year, group in data.groupby(data.index.year):
        enough = len(group) >= config.MIN_ROWS_FOR_STATS
        coin_returns = group.iloc[:, 0]
        rows[int(year)] = {
            "corr_inflation": coin_returns.corr(group["T10YIE_chg"]) if enough else np.nan,
            "corr_dollar": coin_returns.corr(group["DTWEXBGS_ret"]) if enough else np.nan,
            "n": len(group),
        }
    table = pd.DataFrame.from_dict(rows, orient="index", columns=["corr_inflation", "corr_dollar", "n"])
    table.index.name = "year"
    return table


# ---------- question 3: CPI release-day event study (hourly data) ----------

def release_times_utc(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Each date at the release time in New York, converted to UTC (daylight saving is handled by the zone)."""
    local = pd.DatetimeIndex(pd.to_datetime(dates)).tz_localize(None).normalize() + pd.Timedelta(config.RELEASE_TIME + ":00")
    return local.tz_localize(config.RELEASE_TZ).tz_convert("UTC")


def _complete_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    """The hourly candles on a gap-free hourly index; hours with no candle become NaN rows."""
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    return hourly.reindex(full_index)


def hourly_abs_returns(hourly: pd.DataFrame) -> pd.Series:
    """|log(close[t] / close[t - 1 hour])| stamped at candle time t.

    The index is made complete first, so a return that touches a missing candle is NaN
    (it is never computed across the gap).
    """
    close = _complete_hourly(hourly)["close"]
    return np.log(close).diff().abs()


def _column_means(values: np.ndarray, axis: int) -> np.ndarray:
    """Mean over the given axis ignoring NaN; NaN (without a warning) where every value is NaN."""
    present = ~np.isnan(values)
    counts = present.sum(axis=axis)
    totals = np.where(present, values, 0.0).sum(axis=axis)
    return np.where(counts > 0, totals / np.maximum(counts, 1), np.nan)


def event_matrix(abs_ret: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Rows = dates, columns = relative hours -EVENT_HOURS_BEFORE..EVENT_HOURS_AFTER.

    Hour 0 is the candle containing the release moment (08:30 New York). Each cell is looked up
    by timestamp, so a missing candle gives NaN instead of shifting the other hours.
    """
    hours = list(range(-config.EVENT_HOURS_BEFORE, config.EVENT_HOURS_AFTER + 1))
    releases = release_times_utc(dates)
    complete_index = pd.date_range(abs_ret.index.min(), abs_ret.index.max(), freq="h")
    # Match each release to the latest candle starting at or before it, at most 59 minutes earlier
    # (the candle that contains it). A release outside the data has no such candle and gets NaN.
    # merge_asof needs sorted input, so sort and remember each release's original row.
    releases_df = pd.DataFrame({"release": releases, "row": range(len(releases))}).sort_values("release")
    candles_df = pd.DataFrame({"candle": complete_index})
    matched = pd.merge_asof(releases_df, candles_df, left_on="release", right_on="candle",
                            direction="backward", tolerance=pd.Timedelta("59min"))
    hour0 = pd.DatetimeIndex(matched.sort_values("row")["candle"])  # back in the caller's order

    offsets = pd.to_timedelta(hours, unit="h")
    values = np.full((len(dates), len(hours)), np.nan)
    for row, start in enumerate(hour0):
        if pd.isna(start):
            continue  # release outside the data: the whole row stays NaN
        values[row] = abs_ret.reindex(start + offsets).to_numpy()
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates), columns=hours)


def _window_inside_data(dates: pd.DatetimeIndex, first: pd.Timestamp, last: pd.Timestamp) -> np.ndarray:
    """True for each date whose whole event window (hour -6 to hour 12) lies between first and last candle."""
    hour0 = release_times_utc(dates).floor("h")
    start = hour0 - pd.Timedelta(hours=config.EVENT_HOURS_BEFORE)
    end = hour0 + pd.Timedelta(hours=config.EVENT_HOURS_AFTER)
    return np.asarray((start >= first) & (end <= last))


def _ordinary_days(first: pd.Timestamp, last: pd.Timestamp, release_dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Weekdays (New York dates) that are not release dates and whose event window lies inside the data."""
    weekdays = pd.bdate_range(first.tz_convert(config.RELEASE_TZ).tz_localize(None).normalize(),
                              last.tz_convert(config.RELEASE_TZ).tz_localize(None).normalize())
    weekdays = weekdays[~weekdays.isin(release_dates)]
    return weekdays[_window_inside_data(weekdays, first, last)]


def daily_event_comparison(hourly: pd.DataFrame, release_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Realized volatility and volume of CPI days versus other weekdays.

    A day is a New York calendar date. Realized volatility = standard deviation of that day's
    hourly log returns; volume = the day's total hourly volume. Only weekdays whose whole New York
    day lies inside the data are used, and days with fewer than MIN_HOURLY_RETURNS_PER_DAY hourly
    returns are skipped.
    """
    full = _complete_hourly(hourly)
    log_ret = np.log(full["close"]).diff()
    ny_date = full.index.tz_convert(config.RELEASE_TZ).tz_localize(None).normalize()
    per_day = pd.DataFrame({"vol": log_ret.groupby(ny_date).std(ddof=1),
                            "n_returns": log_ret.groupby(ny_date).count(),
                            "volume": full["volume"].groupby(ny_date).sum()})

    # Keep weekdays whose whole New York day (midnight to midnight) is inside the data.
    day_start = per_day.index.tz_localize(config.RELEASE_TZ)
    next_start = (per_day.index + pd.Timedelta(days=1)).tz_localize(config.RELEASE_TZ)
    inside = (day_start >= full.index[0]) & (next_start - pd.Timedelta(hours=1) <= full.index[-1])
    keep = inside & (per_day.index.day_of_week < 5) & (per_day["n_returns"] >= config.MIN_HOURLY_RETURNS_PER_DAY).to_numpy()
    per_day = per_day[keep]
    per_day["cpi_day"] = per_day.index.isin(pd.DatetimeIndex(release_dates).tz_localize(None).normalize())

    grouped = per_day.groupby("cpi_day")
    table = pd.DataFrame({"realized_vol_mean": grouped["vol"].mean(),
                          "realized_vol_std": grouped["vol"].std(ddof=1),
                          "volume_mean": grouped["volume"].mean(),
                          "n_days": grouped["vol"].count()})
    table = table.reindex([False, True])
    table["n_days"] = table["n_days"].fillna(0).astype(int)
    table.index.name = "cpi_day"
    return table


def event_window_profile(hourly: pd.DataFrame, release_dates: pd.DatetimeIndex,
                         draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED) -> pd.DataFrame:
    """Mean absolute return around CPI releases per relative hour, with a bootstrap baseline.

    The baseline repeats `draws` times: pick as many ordinary days as there are CPI days (with
    replacement) and average their event windows. base_low / base_high are the BOOTSTRAP_BAND
    percentiles (2.5% and 97.5%) of those averages.
    """
    abs_ret = hourly_abs_returns(hourly)
    first, last = abs_ret.index[0], abs_ret.index[-1]
    cpi_dates = pd.DatetimeIndex(release_dates)
    cpi_dates = cpi_dates[_window_inside_data(cpi_dates, first, last)]  # CPI days must be comparable to ordinary days
    ordinary = _ordinary_days(first, last, release_dates)
    if len(cpi_dates) == 0 or len(ordinary) == 0:
        raise ValueError("need at least one release date and one ordinary day with a full event window in the data")

    cpi_matrix = event_matrix(abs_ret, cpi_dates)
    ordinary_matrix = event_matrix(abs_ret, ordinary).to_numpy()  # built once, rows are drawn from it

    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(ordinary), size=(draws, len(cpi_dates)))
    draw_means = _column_means(ordinary_matrix[picks], axis=1)  # shape (draws, hours)
    low, high = np.percentile(draw_means, list(config.BOOTSTRAP_BAND), axis=0)

    profile = pd.DataFrame({"cpi_mean": _column_means(cpi_matrix.to_numpy(), axis=0),
                            "base_mean": _column_means(draw_means, axis=0),
                            "base_low": low, "base_high": high}, index=cpi_matrix.columns)
    profile.index.name = "rel_hour"
    return profile


def reaction_hours(profile: pd.DataFrame) -> int | None:
    """How many hours the CPI reaction lasts: None if hour 0 is not above the baseline band.

    Otherwise the first hour h >= 1 where cpi_mean is back at or below base_high (the reaction
    lasted h hours), or EVENT_HOURS_AFTER + 1 if it never comes back.
    """
    if not profile.loc[0, "cpi_mean"] > profile.loc[0, "base_high"]:
        return None
    for hour in range(1, config.EVENT_HOURS_AFTER + 1):
        if profile.loc[hour, "cpi_mean"] <= profile.loc[hour, "base_high"]:
            return hour
    return config.EVENT_HOURS_AFTER + 1
