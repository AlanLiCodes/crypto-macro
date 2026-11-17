"""Turn the raw JSON files saved by acquire.py into pandas objects.

This module only reads files. It does not import acquire (which would pull in requests).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cryptomacro import config

CANDLE_COLUMNS = ["time", "low", "high", "open", "close", "volume"]  # order used by the Coinbase API


class DataMissingError(Exception):
    """Raised when the raw files a loader needs are not on disk."""


def _missing(path: Path) -> DataMissingError:
    return DataMissingError(f"No data found at {path}. Run: python -m cryptomacro.acquire")


def load_candles(product: str, granularity: int, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """All saved candles for one product and granularity, oldest first, one row per timestamp."""
    folder = raw_dir / "coinbase"
    # Window file names end in the window start (YYYYmmddTHHMM), so sorting by name is oldest first.
    pattern = f"{product}_{granularity}_*.json"
    files = sorted(folder.glob(pattern))
    if not files:
        raise DataMissingError(f"No {pattern} files found in {folder}. Run: python -m cryptomacro.acquire")
    rows = [row for path in files for row in json.loads(path.read_text())]  # an empty window adds nothing

    df = pd.DataFrame(rows, columns=CANDLE_COLUMNS).astype(float)
    # Neighbouring windows can both contain a boundary candle; keep the first copy.
    df = df.drop_duplicates("time").sort_values("time")
    df.index = pd.to_datetime(df.pop("time"), unit="s", utc=True).rename("time")
    return df


def load_fred_series(series_id: str, raw_dir: Path = config.RAW_DIR) -> pd.Series:
    """One FRED series as floats indexed by date. FRED writes "." for a missing value; it becomes NaN."""
    path = raw_dir / "fred" / f"{series_id}.json"
    if not path.exists():
        raise _missing(path)
    observations = json.loads(path.read_text())["observations"]
    dates = pd.to_datetime([obs["date"] for obs in observations])
    values = [float("nan") if obs["value"] == "." else float(obs["value"]) for obs in observations]
    return pd.Series(values, index=dates.rename("date"), name=series_id, dtype=float)


def split_cpi_release_dates(dates) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Keep at most one release date per calendar month; return (kept, dropped).

    CPI is published once a month. If a month lists two or more dates, the latest one is kept
    and the others are returned as dropped so the notebook can show them.
    """
    dates = pd.DatetimeIndex(dates).unique().sort_values()
    month = dates.to_period("M")
    # The dates are sorted, so the last date of each month is that month's latest one.
    is_kept = ~month.duplicated(keep="last")
    return dates[is_kept], dates[~is_kept]


def _cpi_dates_in_period(raw_dir: Path) -> pd.DatetimeIndex:
    """All saved CPI release dates inside the study period."""
    path = raw_dir / "fred" / "cpi_release_dates.json"
    if not path.exists():
        raise _missing(path)
    entries = json.loads(path.read_text())["release_dates"]
    dates = pd.DatetimeIndex([entry["date"] for entry in entries])
    in_period = (dates >= pd.Timestamp(config.PERIOD_START)) & (dates <= pd.Timestamp(config.PERIOD_END))
    return dates[in_period]


def load_cpi_release_dates(raw_dir: Path = config.RAW_DIR) -> pd.DatetimeIndex:
    """CPI release dates inside the study period, sorted, without duplicates, at most one per month."""
    kept, _ = split_cpi_release_dates(_cpi_dates_in_period(raw_dir))
    return kept


def load_dropped_cpi_dates(raw_dir: Path = config.RAW_DIR) -> pd.DatetimeIndex:
    """Release dates left out because a later date in the same month was kept."""
    _, dropped = split_cpi_release_dates(_cpi_dates_in_period(raw_dir))
    return dropped


def log_returns(close: np.ndarray) -> np.ndarray:
    """Log return between each pair of consecutive values; one element shorter than the input."""
    return np.log(close[1:] / close[:-1])


def label_regimes(dff: pd.Series) -> pd.Series:
    """Label each calendar day "cutting", "holding" or "raising" from the 90-day change in the fed funds rate.

    The raw DFF series has gaps (weekends, holidays). It is forward-filled onto every calendar day
    here, and only here, so that "90 days earlier" always has a value. The first 90 days of the
    series are NaN, which is why acquire.py downloads DFF from 100 days before the study period.
    """
    daily = dff.reindex(pd.date_range(dff.index.min(), dff.index.max(), freq="D")).ffill()
    change = daily - daily.shift(config.REGIME_LOOKBACK_DAYS)
    # Right-closed bins: a change of exactly -0.125 is "cutting". The middle bin stops just below
    # +0.125, so a change of exactly +0.125 is "raising".
    bins = [-np.inf, -config.REGIME_THRESHOLD, np.nextafter(config.REGIME_THRESHOLD, 0), np.inf]
    regime = pd.cut(change, bins=bins, labels=config.REGIME_LABELS)
    return regime.rename("regime")


def _daily_close(product: str, raw_dir: Path) -> pd.Series:
    """Daily candle close, labelled by the candle's UTC date (tz-naive)."""
    close = load_candles(product, config.GRANULARITIES["daily"], raw_dir)["close"]
    return close.set_axis(close.index.tz_localize(None).normalize().rename("date"))


def _consecutive_log_returns(levels: pd.Series) -> pd.Series:
    """Log return between consecutive rows. NaN when either row is NaN; the first row is NaN."""
    return pd.Series(np.concatenate([[np.nan], log_returns(levels.to_numpy())]), index=levels.index)


def build_daily_panel(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame:
    """One row per US trading day (a day with an S&P 500 value): levels, changes and the rate regime.

    Crypto trades every day, so a Monday change covers the whole weekend. Nothing is forward-filled:
    a missing value stays NaN and makes the changes on either side of it NaN.
    """
    fred = {series_id: load_fred_series(series_id, raw_dir) for series_id in config.FRED_SERIES}
    trading_days = fred["SP500"].dropna().index.rename("date")

    panel = pd.DataFrame(index=trading_days)
    for coin, product in config.PRODUCTS.items():
        panel[coin] = _daily_close(product, raw_dir).reindex(trading_days)
    for series_id, series in fred.items():
        panel[series_id] = series.reindex(trading_days)

    for column in ["BTC", "ETH", "SP500", "NASDAQCOM"]:
        panel[f"{column}_ret"] = _consecutive_log_returns(panel[column])
    for column in ["DGS10", "DFF", "T10YIE"]:
        panel[f"{column}_chg"] = panel[column].diff()
    panel["DTWEXBGS_ret"] = panel["DTWEXBGS"].pct_change(fill_method=None)

    panel["regime"] = label_regimes(fred["DFF"]).reindex(trading_days)
    return panel
