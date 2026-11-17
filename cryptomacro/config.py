"""Every tunable number and name for the project lives here."""
from pathlib import Path

# Paths resolve from the project root, not the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
# Optional file holding the line FRED_API_KEY=...; git-ignored, never committed.
ENV_FILE = PROJECT_ROOT / ".env"

PERIOD_START = "2020-01-01"
PERIOD_END = "2026-09-30"

PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
GRANULARITIES = {"daily": 86400, "hourly": 3600}  # seconds per candle

FRED_SERIES = {"SP500": "S&P 500", "NASDAQCOM": "Nasdaq Composite",
               "DGS10": "10-year Treasury yield", "DFF": "Fed funds rate",
               "T10YIE": "10-year breakeven inflation",
               "DTWEXBGS": "Trade-weighted dollar index"}
CHANGE_COLUMN = {"SP500": "SP500_ret", "NASDAQCOM": "NASDAQCOM_ret", "DGS10": "DGS10_chg",
                 "DFF": "DFF_chg", "T10YIE": "T10YIE_chg", "DTWEXBGS": "DTWEXBGS_ret"}
CPI_RELEASE_ID = 10
# Written by acquire.main only after every download succeeded; the notebook refuses to load without it.
PULL_COMPLETE_FILE = "pull_complete.json"

COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
FRED_URL = "https://api.stlouisfed.org/fred"
MAX_CANDLES = 300
COINBASE_PAUSE = 0.25
FRED_PAUSE = 1.0

MAX_TRIES = 5
TIMEOUT = 30
BACKOFF_BASE = 1.0

REGIME_LOOKBACK_DAYS = 90
# DFF is downloaded this many days before PERIOD_START so the first days of the period have a regime.
REGIME_FETCH_MARGIN_DAYS = 100
REGIME_THRESHOLD = 0.125
REGIME_LABELS = ["cutting", "holding", "raising"]  # from the lowest 90-day DFF change to the highest
EVENT_HOURS_BEFORE = 6
EVENT_HOURS_AFTER = 12
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_BAND = (2.5, 97.5)  # percentiles of the bootstrap averages shown as the baseline band
MIN_HOURLY_RETURNS_PER_DAY = 12  # a day with fewer hourly returns is left out of the CPI-day comparison
SEED = 510
MIN_ROWS_FOR_STATS = 3  # fewer complete rows than this and a regression or correlation is not reported
DEFAULT_WINDOW = 90
# A rolling window needs at least this share of complete days, so one bond-market holiday
# (stocks open, Treasuries closed) does not blank out every window that contains it.
MIN_WINDOW_FRACTION = 0.8
WINDOW_MIN = 30
WINDOW_MAX = 180

CRYPTO_DAYS_PER_YEAR = 365  # crypto trades every calendar day
STOCK_DAYS_PER_YEAR = 252  # US trading days in a year
WEEKLY_RULE = "W-FRI"  # weekly levels are the last value of each week ending on Friday

RELEASE_TIME = "08:30"
RELEASE_TZ = "America/New_York"
