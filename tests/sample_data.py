"""Fake raw data in exactly the shapes acquire.py saves, so tests never need the network.

write_sample_raw builds a small world with these properties (later tests rely on them):
  * Trading days are the weekdays minus two holidays, which are "." in every FRED series.
  * On the bond-market-only holidays (Columbus Day, Veterans Day) stocks trade, but DGS10, T10YIE
    and DTWEXBGS are "."; SP500, NASDAQCOM and DFF keep their values. No CPI date falls on them.
  * Crypto trades every calendar day. Daily candles start at 00:00 UTC; hourly ones on the hour.
  * On trading days, BTC's log return since the previous trading day's close is
    0.8 x the S&P 500 return + small macro effects + small noise (ETH: similar, noisier).
  * Weekend/holiday crypto moves are small.
  * Each day's daily close is the last hourly close of that UTC day.
  * On CPI release days, the 3 hourly candles starting with the one that contains 08:30
    New York time move about 3x as much and trade more volume.
  * DFF rises during 2022 and falls in late 2023, so all three rate regimes occur. Like the real
    download, the DFF file starts REGIME_FETCH_MARGIN_DAYS before `start` (at its starting level),
    so every panel day has a regime label.
  * The pull-complete marker is written last, as acquire.main does.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from cryptomacro import config
from cryptomacro.acquire import candle_windows, window_path, write_json, write_pull_marker

HOLIDAYS = ["2022-07-04", "2023-11-23"]
BOND_ONLY_HOLIDAYS = ["2022-10-10", "2022-11-11", "2023-10-09", "2023-11-10"]
BOND_SERIES = ["DGS10", "T10YIE", "DTWEXBGS"]  # the series with no value on a bond-only holiday
# (date, change in the fed funds rate in percentage points)
DFF_STEPS = [("2022-03-17", 0.25), ("2022-05-05", 0.50), ("2022-06-16", 0.75), ("2022-07-28", 0.75),
             ("2022-09-22", 0.75), ("2022-11-03", 0.75), ("2022-12-15", 0.50), ("2023-02-02", 0.25),
             ("2023-03-23", 0.25), ("2023-05-04", 0.25),
             ("2023-11-30", -0.25), ("2023-12-14", -0.25)]

START_PRICE = {"BTC": 40000.0, "ETH": 3000.0}
START_VOLUME = {"BTC": 100.0, "ETH": 1000.0}
# How strongly each coin follows the S&P 500, and how much idiosyncratic noise it has per trading day.
SP_BETA = {"BTC": 0.8, "ETH": 0.9}
NOISE_SD = {"BTC": 0.003, "ETH": 0.006}
# Return effect per unit change: yields down, breakeven inflation up, dollar down.
DGS10_BETA, T10YIE_BETA, DOLLAR_BETA = -0.05, 0.15, -0.75
WEEKEND_SD = 0.002  # crypto barely moves between two trading days' closes
HOUR_SD = 0.003  # ordinary hourly log-return noise
CPI_MULTIPLIER = 3  # hourly moves are this much larger on the 3 candles around a release
CPI_VOLUME_MULTIPLIER = 2.5
CPI_CANDLES = 3


# ---------- calendar ----------

def make_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Weekdays between start and end, minus the holidays."""
    weekdays = pd.bdate_range(start, end)
    return weekdays[~weekdays.isin(pd.to_datetime(HOLIDAYS))]


def make_cpi_dates(start: str, end: str, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """One release per month: the first trading day on or after the 12th."""
    dates = []
    for month_start in pd.date_range(pd.Timestamp(start).replace(day=1), end, freq="MS"):
        on_or_after = trading_days[trading_days >= month_start + pd.Timedelta(days=11)]
        if len(on_or_after) and on_or_after[0] <= pd.Timestamp(end) and on_or_after[0].month == month_start.month:
            dates.append(on_or_after[0])
    return pd.DatetimeIndex(dates)


def release_candle_positions(cpi_dates: pd.DatetimeIndex, first_hour: pd.Timestamp) -> np.ndarray:
    """Positions (hours since first_hour) of the release candle and the next two, for every release."""
    positions = []
    for date in cpi_dates:
        # 08:30 New York is 12:30 or 13:30 UTC depending on daylight saving; the candle starts on the hour.
        release = pd.Timestamp(f"{date:%Y-%m-%d} {config.RELEASE_TIME}", tz=config.RELEASE_TZ)
        first = (release.tz_convert("UTC").floor("h") - first_hour) // pd.Timedelta(hours=1)
        positions.extend(range(first, first + CPI_CANDLES))
    return np.array(positions, dtype=int)


# ---------- macro series (trading days only) ----------

def rounded_path(start_level: float, changes: np.ndarray, decimals: int, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Turn random changes into a level path, round it like FRED does, and return (levels, changes).

    The returned changes are recomputed from the rounded levels, so a test that differences the
    saved values sees exactly the changes the crypto returns were built from.
    kind "log" means percentage-style moves (levels multiply); "diff" means levels add.
    """
    if kind == "log":
        levels = np.round(start_level * np.exp(np.cumsum(changes)), decimals)
        return levels, np.diff(np.log(levels), prepend=np.log(levels[0]))
    levels = np.round(start_level + np.cumsum(changes), decimals)
    return levels, np.diff(levels, prepend=levels[0])


def make_macro(trading_days: pd.DatetimeIndex, rng: np.random.Generator) -> dict:
    """Levels for the six FRED series and the changes the crypto returns react to."""
    n = len(trading_days)
    sp, sp_chg = rounded_path(4000, rng.normal(0, 0.010, n), 2, "log")
    # Nasdaq moves with the S&P but is a bit more volatile.
    nasdaq, _ = rounded_path(12000, 1.2 * sp_chg + rng.normal(0, 0.004, n), 2, "log")
    dgs10, dgs10_chg = rounded_path(1.6, rng.normal(0, 0.05, n), 2, "diff")
    t10yie, t10yie_chg = rounded_path(2.3, rng.normal(0, 0.02, n), 2, "diff")
    dollar, dollar_ret = rounded_path(120, rng.normal(0, 0.004, n), 2, "log")

    # The fed funds rate is flat between meetings and jumps on the step dates.
    dff = np.full(n, 0.08)
    for date, step in DFF_STEPS:
        dff[trading_days >= pd.Timestamp(date)] += step

    return {"levels": {"SP500": sp, "NASDAQCOM": nasdaq, "DGS10": dgs10, "DFF": dff,
                       "T10YIE": t10yie, "DTWEXBGS": dollar},
            "sp_ret": sp_chg, "dgs10_chg": dgs10_chg, "t10yie_chg": t10yie_chg, "dollar_ret": dollar_ret}


# ---------- crypto ----------

def calendar_day_returns(days: pd.DatetimeIndex, trading_days: pd.DatetimeIndex,
                         trading_targets: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One log return per calendar day.

    trading_targets[i] is the log return wanted from the previous trading day's close to trading
    day i's close. Non-trading days get small random returns, and the next trading day's own
    return is reduced by them, so the return across the whole span equals the target exactly.
    """
    target_by_day = dict(zip(trading_days, trading_targets))
    returns = []
    pending = 0.0  # returns of non-trading days since the last trading day
    for day in days:
        if day in target_by_day:
            returns.append(target_by_day[day] - pending)
            pending = 0.0
        else:
            returns.append(rng.normal(0, WEEKEND_SD))
            pending += returns[-1]
    return np.array(returns)


def make_hourly_candles(coin: str, day_returns: np.ndarray, cpi_positions: np.ndarray,
                        first_hour: pd.Timestamp, rng: np.random.Generator) -> pd.DataFrame:
    """Hourly OHLCV whose 24 returns each day add up to that day's return."""
    n_days = len(day_returns)
    n_hours = n_days * 24
    noise = rng.normal(0, HOUR_SD, n_hours)
    noise[cpi_positions] *= CPI_MULTIPLIER
    # Remove each day's mean so the noise cancels within the day; the day's return is then spread evenly.
    by_day = noise.reshape(n_days, 24)
    noise = by_day - by_day.mean(axis=1, keepdims=True)
    hourly_ret = (noise + day_returns[:, None] / 24).ravel()

    close = np.round(START_PRICE[coin] * np.exp(np.cumsum(hourly_ret)), 2)
    open_ = np.concatenate([[START_PRICE[coin]], close[:-1]])  # each candle opens at the last close
    high = np.round(np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.001, n_hours))), 2)
    low = np.round(np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.001, n_hours))), 2)
    volume = START_VOLUME[coin] * rng.lognormal(0, 0.3, n_hours)
    volume[cpi_positions] *= CPI_VOLUME_MULTIPLIER
    index = pd.date_range(first_hour, periods=n_hours, freq="h")
    return pd.DataFrame({"low": low, "high": high, "open": open_, "close": close,
                         "volume": np.round(volume, 4)}, index=index)


def hourly_to_daily(hourly: pd.DataFrame) -> pd.DataFrame:
    """Daily candles (00:00 UTC) built from the hourly ones, so the two are consistent."""
    return hourly.resample("D").agg({"low": "min", "high": "max", "open": "first",
                                     "close": "last", "volume": "sum"})


# ---------- writing files ----------

def pick_missing_hour(hourly_index: pd.DatetimeIndex, cpi_dates: pd.DatetimeIndex,
                      rng: np.random.Generator) -> pd.Timestamp:
    """A random hour at least 3 days away from every CPI release (so event studies are unaffected)."""
    release_days = cpi_dates.tz_localize("UTC")
    # Distance from each day's 00:00 to the nearest release day. Needing 4 days (not 3) leaves
    # room for the hour within the day and for the release time within its day.
    distance = [abs(day_start - release_days).min() for day_start in hourly_index[::24]]
    far_days = [i for i, d in enumerate(distance) if d >= pd.Timedelta(days=4)]
    day = rng.choice(far_days[1:-1])  # not the first or last day, which sit on window edges
    return hourly_index[day * 24 + rng.integers(0, 24)]


def write_candle_windows(raw_dir: Path, product: str, granularity: int, candles: pd.DataFrame) -> None:
    """Save candles in window files laid out like acquire.fetch_candles does (newest first)."""
    # [time, low, high, open, close, volume] rows, oldest first
    rows = [[int(t.timestamp()), *values] for t, values in
            zip(candles.index, candles[["low", "high", "open", "close", "volume"]].to_numpy().tolist())]
    times = [row[0] for row in rows]
    start = candles.index[0].to_pydatetime()
    end = candles.index[-1].to_pydatetime()
    previous_window = []
    for number, (window_start, window_end) in enumerate(candle_windows(start, end, granularity)):
        low, high = window_start.timestamp(), window_end.timestamp()
        window_rows = [row for row, t in zip(rows, times) if low <= t <= high]
        saved = window_rows[::-1]  # the API returns newest first
        if number == 1 and previous_window:
            saved.append(previous_window[-1])  # the second window repeats the first window's last candle
        write_json(window_path(raw_dir, product, granularity, window_start), saved)
        previous_window = window_rows


def write_fred_files(raw_dir: Path, trading_days: pd.DatetimeIndex, weekdays: pd.DatetimeIndex,
                     macro: dict, cpi_dates: pd.DatetimeIndex) -> None:
    # DFF also covers the weekdays in the margin before the sample, at its starting level.
    margin_start = weekdays[0] - pd.Timedelta(days=config.REGIME_FETCH_MARGIN_DAYS)
    margin_days = pd.bdate_range(margin_start, weekdays[0] - pd.Timedelta(days=1))
    for series_id, levels in macro["levels"].items():
        value_by_day = dict(zip(trading_days, levels))
        days = weekdays
        if series_id == "DFF":
            value_by_day.update({day: levels[0] for day in margin_days})
            days = margin_days.append(weekdays)
        if series_id in BOND_SERIES:
            for day in pd.to_datetime(BOND_ONLY_HOLIDAYS):
                value_by_day.pop(day, None)  # bond market closed: this day becomes "."
        # FRED has a row for every weekday; the holidays have no value and show ".".
        observations = [{"date": f"{day:%Y-%m-%d}",
                         "value": f"{value_by_day[day]:.2f}" if day in value_by_day else "."}
                        for day in days]
        write_json(raw_dir / "fred" / f"{series_id}.json", {"observations": observations})
    write_json(raw_dir / "fred" / "cpi_release_dates.json",
               {"release_dates": [{"release_id": config.CPI_RELEASE_ID, "date": f"{d:%Y-%m-%d}"}
                                  for d in cpi_dates]})


def write_sample_raw(raw_dir: Path, start: str = "2022-01-03", end: str = "2023-12-29", seed: int = 0) -> None:
    """Write fake raw files for both coins, both granularities, six FRED series and CPI dates."""
    rng = np.random.default_rng(seed)
    weekdays = pd.bdate_range(start, end)
    trading_days = make_trading_days(start, end)
    cpi_dates = make_cpi_dates(start, end, trading_days)
    macro = make_macro(trading_days, rng)

    days = pd.date_range(start, end, freq="D")  # crypto trades every calendar day
    first_hour = pd.Timestamp(start, tz="UTC")
    cpi_positions = release_candle_positions(cpi_dates, first_hour)
    # Effects of the macro series on crypto returns (the same for both coins).
    macro_effect = (DGS10_BETA * macro["dgs10_chg"] + T10YIE_BETA * macro["t10yie_chg"]
                    + DOLLAR_BETA * macro["dollar_ret"])

    write_fred_files(raw_dir, trading_days, weekdays, macro, cpi_dates)
    for coin, product in config.PRODUCTS.items():
        targets = (SP_BETA[coin] * macro["sp_ret"] + macro_effect
                   + rng.normal(0, NOISE_SD[coin], len(trading_days)))
        day_returns = calendar_day_returns(days, trading_days, targets, rng)
        hourly = make_hourly_candles(coin, day_returns, cpi_positions, first_hour, rng)
        daily = hourly_to_daily(hourly).tz_convert("UTC")
        daily.index.name = hourly.index.name = None
        # Daily candles are complete; the hourly file lacks one candle, as real Coinbase data sometimes does.
        hourly = hourly.drop(pick_missing_hour(hourly.index, cpi_dates, rng))
        write_candle_windows(raw_dir, product, config.GRANULARITIES["daily"], daily)
        write_candle_windows(raw_dir, product, config.GRANULARITIES["hourly"], hourly)
    write_pull_marker(raw_dir)
