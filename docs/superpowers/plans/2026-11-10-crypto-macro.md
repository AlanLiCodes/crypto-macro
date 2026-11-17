# Crypto and the Macro Economy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the DSCI 510 project: fetch Coinbase and FRED data to cached JSON, answer four questions with pandas/NumPy, and present them in a widget notebook.

**Architecture:** A small package `cryptomacro/` with one module per stage (acquire → clean → analysis → plots → dashboard). Only `acquire.py` touches the network; everything else reads `data/raw/`. The notebook imports the package and shows the dashboard plus one section per question.

**Tech Stack:** Python 3.10+, requests, pandas, numpy, matplotlib, ipywidgets, jupyter, pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-crypto-macro-design.md`

## Global Constraints

- Period: 2020-01-01 to 2026-09-30.
- Only `cryptomacro/acquire.py` imports `requests`. No test touches the network.
- FRED key is read from the `FRED_API_KEY` environment variable only; never written to any file.
- `data/` and `.env` are git-ignored; no raw values are committed.
- Coinbase: at most 300 candles per request, 0.25 s pause between calls. FRED: 1 s pause.
- Retry: up to 5 tries, exponential backoff, on HTTP 429 and 5xx only.
- All paths resolve from the project root (`config.PROJECT_ROOT`), not the current working directory.
- Plain, commented, explainable code: short functions, no metaclasses, no async, no third-party helpers beyond the stack above.
- Every tunable number lives in `config.py`.
- One deviation from the spec's layout: widget wiring lives in `cryptomacro/dashboard.py` so it can be smoke-tested; the notebook calls it.

## Review Focus

1. **Interrupted pull leaves a half-written cache file** → the next run must refetch it, not crash or silently load it. (Task 2)
2. **Missing hourly candles (exchange outages)** → the event study must place hours by timestamp, never by row position; a missing hour is NaN, not a shifted hour. (Task 8)
3. **Running from another directory, or opening the notebook before/after a partial pull** → paths still resolve, and a missing file gives one message naming the file and the command to run. (Tasks 4, 9)
4. **Holiday mismatches and gaps in FRED series** (stock market closed but bond market open, leading NaNs) → the panel has no inf values and no row where a return spans a NaN as if it were zero. (Task 5)
5. **Too few observations** (a regime with under 3 days, a window longer than the data) → the result is NaN with the count shown, not an exception or a meaningless slope. (Tasks 6, 7)

---

## File Structure

```
crypto-macro/
  README.md  requirements.txt  .gitignore  pytest.ini
  cryptomacro/__init__.py  config.py  acquire.py  clean.py  analysis.py  plots.py  dashboard.py
  crypto_macro.ipynb
  tools/build_notebook.py        writes crypto_macro.ipynb with nbformat
  tests/sample_data.py           writes synthetic raw JSON in the real API shapes
  tests/test_acquire.py  test_clean.py  test_analysis.py  test_event_study.py  test_dashboard.py
```

---

### Task 1: Skeleton, config and `get_json`

**Files:**
- Create: `requirements.txt`, `.gitignore`, `pytest.ini`, `cryptomacro/__init__.py`, `cryptomacro/config.py`, `cryptomacro/acquire.py`
- Test: `tests/test_acquire.py`

**Interfaces — Produces:**

`config.py` constants (exact names and values):

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PERIOD_START = "2020-01-01"; PERIOD_END = "2026-09-30"
PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
GRANULARITIES = {"daily": 86400, "hourly": 3600}
FRED_SERIES = {"SP500": "S&P 500", "NASDAQCOM": "Nasdaq Composite",
               "DGS10": "10-year Treasury yield", "DFF": "Fed funds rate",
               "T10YIE": "10-year breakeven inflation",
               "DTWEXBGS": "Trade-weighted dollar index"}
CHANGE_COLUMN = {"SP500": "SP500_ret", "NASDAQCOM": "NASDAQCOM_ret", "DGS10": "DGS10_chg",
                 "DFF": "DFF_chg", "T10YIE": "T10YIE_chg", "DTWEXBGS": "DTWEXBGS_ret"}
CPI_RELEASE_ID = 10
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
FRED_URL = "https://api.stlouisfed.org/fred"
MAX_CANDLES = 300; COINBASE_PAUSE = 0.25; FRED_PAUSE = 1.0
MAX_TRIES = 5; TIMEOUT = 30; BACKOFF_BASE = 1.0
REGIME_LOOKBACK_DAYS = 90; REGIME_THRESHOLD = 0.125
EVENT_HOURS_BEFORE = 6; EVENT_HOURS_AFTER = 12
BOOTSTRAP_DRAWS = 2000; SEED = 510
DEFAULT_WINDOW = 90; WINDOW_MIN = 30; WINDOW_MAX = 180
RELEASE_TIME = "08:30"; RELEASE_TZ = "America/New_York"
```

`acquire.py`:

```python
class AcquisitionError(Exception): ...
def get_json(url, params=None, *, session=None, pause=0.0,
             max_tries=config.MAX_TRIES, timeout=config.TIMEOUT, sleep=time.sleep): ...
```

`get_json` calls `session.get(url, params=params, timeout=timeout)` (a new `requests.Session` when `session` is None). On status 429 or ≥500 it sleeps `BACKOFF_BASE * 2**attempt` and retries; after `max_tries` failures it raises `AcquisitionError` naming the URL and last status. Other 4xx: `raise_for_status()`. On success it calls `sleep(pause)` when `pause > 0` and returns `response.json()`.

- [ ] **Step 1: Write failing tests** in `tests/test_acquire.py`, with a `FakeSession` helper (takes a list of `(status, payload)` tuples, records every `get` call, returns objects with `.status_code`, `.json()`, `.text`, `.raise_for_status()` raising `requests.HTTPError` for ≥400):

```python
def test_get_json_returns_payload():
    s = FakeSession([(200, {"a": 1})])
    assert get_json("u", {"p": 1}, session=s, sleep=lambda x: None) == {"a": 1}
    assert s.calls[0]["params"] == {"p": 1} and s.calls[0]["timeout"] == 30

def test_get_json_retries_429_and_5xx_with_backoff():
    sleeps = []
    s = FakeSession([(429, {}), (503, {}), (200, [1])])
    assert get_json("u", session=s, sleep=sleeps.append) == [1]
    assert sleeps == [1.0, 2.0]

def test_get_json_gives_up_after_max_tries():
    s = FakeSession([(500, {})] * 5)
    with pytest.raises(AcquisitionError, match="500"):
        get_json("u", session=s, sleep=lambda x: None)
    assert len(s.calls) == 5

def test_get_json_does_not_retry_404():
    s = FakeSession([(404, {})])
    with pytest.raises(requests.HTTPError):
        get_json("u", session=s, sleep=lambda x: None)
    assert len(s.calls) == 1

def test_get_json_pauses_after_success():
    sleeps = []
    get_json("u", session=FakeSession([(200, {})]), pause=0.25, sleep=sleeps.append)
    assert sleeps == [0.25]
```

- [ ] **Step 2:** Create a virtualenv, `pip install -r requirements.txt`, `git init`. Run `pytest tests/test_acquire.py -v`. Expected: FAIL (import error).
- [ ] **Step 3:** Write `config.py` and `get_json` as specified. `pytest.ini` sets `testpaths = tests`. `.gitignore`: `data/`, `.env`, `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.pytest_cache/`.
- [ ] **Step 4:** Run `pytest tests/test_acquire.py -v`. Expected: 5 passed.
- [ ] **Step 5:** Commit: `feat: project skeleton, config, get_json with retry`.

### Task 2: Coinbase paging, caching and resume

**Files:** Modify `cryptomacro/acquire.py`; Test `tests/test_acquire.py`

**Interfaces:**
- Consumes: `get_json`, config constants.
- Produces:

```python
def candle_windows(start: datetime, end: datetime, granularity: int,
                   max_candles: int = config.MAX_CANDLES) -> Iterator[tuple[datetime, datetime]]: ...
def window_path(raw_dir: Path, product: str, granularity: int, start: datetime) -> Path:
    # raw_dir / "coinbase" / f"{product}_{granularity}_{start:%Y%m%dT%H%M}.json"
def write_json(path: Path, payload) -> None      # write to path + ".tmp", then os.replace
def read_json(path: Path)                        # returns None if missing or not valid JSON
def fetch_candles(product: str, granularity: int, start: datetime, end: datetime,
                  raw_dir: Path = config.RAW_DIR, *, session=None,
                  sleep=time.sleep) -> tuple[int, int]:   # (fetched, skipped)
```

`candle_windows` is a generator. Each window covers `max_candles` candle start times: it yields `(w_start, min(w_start + (max_candles-1)*granularity seconds, end))`, and the next window starts at `w_start + max_candles*granularity` seconds. Windows never overlap. `fetch_candles` requests each window with params `{"granularity": granularity, "start": w_start ISO-8601 "Z", "end": w_end ISO-8601 "Z"}`, pause `COINBASE_PAUSE`, saves the response list as-is, and skips windows whose file `read_json` returns non-None for.

- [ ] **Step 1: Write failing tests**

```python
UTC = timezone.utc
def test_candle_windows_daily_sizes():
    w = list(candle_windows(datetime(2020,1,1,tzinfo=UTC), datetime(2021,12,31,tzinfo=UTC), 86400))
    assert w[0] == (datetime(2020,1,1,tzinfo=UTC), datetime(2020,10,26,tzinfo=UTC))  # 300 days inclusive
    assert w[1][0] == datetime(2020,10,27,tzinfo=UTC)
    assert w[-1][1] == datetime(2021,12,31,tzinfo=UTC) and len(w) == 3

def test_candle_windows_hourly_is_twelve_and_a_half_days():
    w = next(candle_windows(datetime(2020,1,1,tzinfo=UTC), datetime(2020,3,1,tzinfo=UTC), 3600))
    assert w[1] - w[0] == timedelta(hours=299)

def test_candle_windows_is_a_generator():
    assert inspect.isgeneratorfunction(candle_windows)

def test_fetch_candles_saves_each_window_and_resumes(tmp_path):
    start, end = datetime(2020,1,1,tzinfo=UTC), datetime(2021,12,31,tzinfo=UTC)
    s = FakeSession([(200, [[1,1,1,1,1,1]])] * 3)
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s, sleep=lambda x: None) == (3, 0)
    assert (tmp_path / "coinbase" / "BTC-USD_86400_20200101T0000.json").exists()
    assert s.calls[0]["params"] == {"granularity": 86400, "start": "2020-01-01T00:00:00Z", "end": "2020-10-26T00:00:00Z"}
    s2 = FakeSession([])
    assert fetch_candles("BTC-USD", 86400, start, end, tmp_path, session=s2, sleep=lambda x: None) == (0, 3)

def test_fetch_candles_keeps_empty_windows(tmp_path):      # an empty list is a valid cached answer
    ...one window, payload [] → file contains [], second run skips it

def test_fetch_candles_refetches_corrupt_file(tmp_path):   # Review Focus 1
    ...pre-write '[[1, 2' to the first window's path → run fetches 1, file is valid JSON afterwards

def test_write_json_leaves_no_tmp_file(tmp_path):
    ...after write_json, only the target file exists in the directory
```

- [ ] **Step 2:** Run `pytest tests/test_acquire.py -v`. Expected: new tests FAIL.
- [ ] **Step 3:** Implement the five functions.
- [ ] **Step 4:** Run tests. Expected: all pass.
- [ ] **Step 5:** Commit: `feat: Coinbase candle paging with cache and resume`.

### Task 3: FRED fetch, key handling and the command-line entry

**Files:** Modify `cryptomacro/acquire.py`; Test `tests/test_acquire.py`

**Interfaces — Produces:**

```python
def fred_api_key() -> str      # strips whitespace; AcquisitionError if unset or empty
def fetch_fred_series(series_id: str, raw_dir: Path = config.RAW_DIR, *, session=None,
                      sleep=time.sleep) -> bool      # True if fetched, False if cached
def fetch_cpi_release_dates(raw_dir: Path = config.RAW_DIR, *, session=None,
                            sleep=time.sleep) -> bool
def main(argv=None) -> int
```

- Series: `GET {FRED_URL}/series/observations` with params `series_id`, `api_key`, `file_type="json"`, `observation_start=PERIOD_START`, `observation_end=PERIOD_END`; saved to `raw_dir/fred/<series_id>.json`.
- Release dates: `GET {FRED_URL}/release/dates` with `release_id=10`, `api_key`, `file_type="json"`, `include_release_dates_with_no_data="true"`, `limit=10000`, `sort_order="asc"`; saved to `raw_dir/fred/cpi_release_dates.json`.
- An HTTP 400 from FRED becomes `AcquisitionError("FRED rejected the request — check that FRED_API_KEY is a valid 32-character key. FRED said: <error_message>")`.
- Missing key message, exact: `"FRED_API_KEY is not set. Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html and run: export FRED_API_KEY=your_key"`.
- `main` fetches daily and hourly candles for both products, then the six series, then the release dates; prints one progress line per product/series and a final `Done: N fetched, M already on disk.`; catches `AcquisitionError`, prints `Error: <message>` to stderr, returns 1. The key is checked before any request. `python -m cryptomacro.acquire` runs `main`.

- [ ] **Step 1: Write failing tests**

```python
def test_fred_api_key_missing(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(AcquisitionError, match="FRED_API_KEY is not set"): fred_api_key()

def test_fred_api_key_strips_whitespace(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", " abc \n"); assert fred_api_key() == "abc"

def test_fetch_fred_series_params_and_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "k")
    s = FakeSession([(200, {"observations": []})])
    assert fetch_fred_series("DGS10", tmp_path, session=s, sleep=lambda x: None) is True
    p = s.calls[0]["params"]
    assert p["series_id"] == "DGS10" and p["observation_start"] == "2020-01-01" and p["observation_end"] == "2026-09-30" and p["file_type"] == "json"
    assert fetch_fred_series("DGS10", tmp_path, session=FakeSession([]), sleep=lambda x: None) is False

def test_fetch_cpi_release_dates_uses_release_10(tmp_path, monkeypatch): ...params["release_id"] == 10

def test_bad_key_gives_clear_error(tmp_path, monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "bad")
    s = FakeSession([(400, {"error_message": "Bad Request. The value for variable api_key is not registered."})])
    with pytest.raises(AcquisitionError, match="check that FRED_API_KEY"): fetch_fred_series("DGS10", tmp_path, session=s, sleep=lambda x: None)

def test_main_without_key_returns_1_and_makes_no_requests(monkeypatch, capsys):
    ...delenv; assert main([]) == 1; "FRED_API_KEY is not set" in capsys.readouterr().err

def test_key_never_written_to_disk(tmp_path, monkeypatch):
    ...setenv "SECRETKEY123"; fetch a series; no file under tmp_path contains "SECRETKEY123"
```

- [ ] **Step 2:** Run tests; expected FAIL. **Step 3:** Implement. **Step 4:** Run; all pass.
- [ ] **Step 5:** Commit: `feat: FRED acquisition, key handling, acquire CLI`.

### Task 4: Loaders and sample data

**Files:** Create `cryptomacro/clean.py`, `tests/sample_data.py`; Test `tests/test_clean.py`

**Interfaces — Produces:**

```python
class DataMissingError(Exception): ...
def load_candles(product: str, granularity: int, raw_dir: Path = config.RAW_DIR) -> pd.DataFrame
    # UTC DatetimeIndex named "time", float columns low, high, open, close, volume;
    # sorted oldest first, duplicate timestamps dropped (keep first)
def load_fred_series(series_id: str, raw_dir: Path = config.RAW_DIR) -> pd.Series
    # tz-naive DatetimeIndex named "date", float values, "." -> NaN, name = series_id
def load_cpi_release_dates(raw_dir: Path = config.RAW_DIR) -> pd.DatetimeIndex
    # tz-naive, sorted, unique, limited to PERIOD_START..PERIOD_END
```

- Candle rows are flattened from all window files with one list comprehension over files and rows.
- When no file is found, raise `DataMissingError` with the message `"No data found at <path>. Run: python -m cryptomacro.acquire"`.
- `tests/sample_data.py` produces `write_sample_raw(raw_dir, start="2022-01-03", end="2023-12-29", seed=0) -> None`. It writes files in the exact API shapes: Coinbase windows (newest first, one duplicated boundary candle, one missing hourly candle) for both products at both granularities; the six FRED series as `{"observations": [{"date": "...", "value": "..."}]}` on weekdays with `"."` on two holidays; and `{"release_dates": [{"release_id": 10, "date": "..."}]}` with one mid-month weekday per month. Prices are seeded random walks. BTC daily returns are built as `0.8 × SP500 return + noise`, so correlations are clearly positive. Hourly BTC absolute returns are 3× larger for the 3 hours starting at the release candle on CPI days.

- [ ] **Step 1: Write failing tests** (hand-written tiny files in `tmp_path`, not the generator):

```python
def test_load_candles_sorts_dedupes_and_types(tmp_path):
    # two window files, newest-first rows, one timestamp repeated across files
    df = load_candles("BTC-USD", 86400, tmp_path)
    assert list(df.columns) == ["low", "high", "open", "close", "volume"]
    assert df.index.is_monotonic_increasing and df.index.is_unique and str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2020-01-01", tz="UTC") and df["close"].iloc[0] == 7200.0

def test_load_candles_tolerates_empty_window(tmp_path): ...a file containing [] alongside a real one
def test_load_candles_missing_raises_with_command(tmp_path):
    with pytest.raises(DataMissingError, match="python -m cryptomacro.acquire"): load_candles("BTC-USD", 86400, tmp_path)
def test_load_fred_series_dot_becomes_nan(tmp_path):
    s = load_fred_series("SP500", tmp_path)   # values "3257.85", ".", "3234.85"
    assert np.isnan(s.iloc[1]) and s.iloc[0] == 3257.85 and s.name == "SP500"
def test_load_cpi_release_dates_limited_to_period(tmp_path):
    # dates 2019-12-11, 2020-01-14, 2026-10-14 -> only 2020-01-14 remains
def test_sample_data_loads(tmp_path):
    write_sample_raw(tmp_path); assert len(load_candles("ETH-USD", 3600, tmp_path)) > 10000
```

- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement loaders and the generator. **Step 4:** Run; pass.
- [ ] **Step 5:** Commit: `feat: raw data loaders and sample data generator`.

### Task 5: Daily panel, returns and regimes

**Files:** Modify `cryptomacro/clean.py`; Test `tests/test_clean.py`

**Interfaces — Produces:**

```python
def log_returns(close: np.ndarray) -> np.ndarray          # np.log(close[1:] / close[:-1])
def label_regimes(dff: pd.Series) -> pd.Series             # categorical: "cutting", "holding", "raising"
def build_daily_panel(raw_dir: Path = config.RAW_DIR) -> pd.DataFrame
```

Panel: tz-naive DatetimeIndex named `date`, one row per US trading day (a day where `SP500` is not NaN). Columns: levels `BTC`, `ETH` (daily candle close, labelled by the candle's UTC date), `SP500`, `NASDAQCOM`, `DGS10`, `DFF`, `T10YIE`, `DTWEXBGS`; changes `BTC_ret`, `ETH_ret`, `SP500_ret`, `NASDAQCOM_ret` (log returns between consecutive panel rows), `DGS10_chg`, `DFF_chg`, `T10YIE_chg` (`diff`), `DTWEXBGS_ret` (`pct_change(fill_method=None)`); and `regime`. A change is NaN whenever either endpoint is NaN. No forward filling.

`label_regimes`: reindex `dff` to every calendar day and forward-fill within that daily series only, subtract the value `REGIME_LOOKBACK_DAYS` days earlier, then `pd.cut` with bins `[-inf, -0.125, nextafter(0.125, 0), inf]` so a change of exactly −0.125 is cutting and exactly +0.125 is raising. The first 90 days are NaN.

- [ ] **Step 1: Write failing tests**

```python
def test_log_returns_matches_formula():
    np.testing.assert_allclose(log_returns(np.array([100., 110., 99.])), [np.log(1.1), np.log(0.9)])

def test_label_regimes_thresholds():
    # daily series: 1.00 for 100 days, then 1.125 for 100 days, then 0.875 for 100 days
    r = label_regimes(dff)
    assert r.iloc[95] == "holding" and r.iloc[150] == "raising" and r.iloc[250] == "cutting"
    assert r.iloc[:90].isna().all()

def test_panel_only_trading_days_and_weekend_folds_into_monday(tmp_path):
    write_sample_raw(tmp_path); p = build_daily_panel(tmp_path)
    assert (p.index.dayofweek < 5).all() and p["SP500"].notna().all()
    fri, mon = p.index[p.index.dayofweek == 4][0], ...next row
    assert p.loc[mon, "BTC_ret"] == pytest.approx(np.log(p.loc[mon, "BTC"] / p.loc[fri, "BTC"]))

def test_panel_has_expected_columns(tmp_path): ...exact column set above

def test_panel_no_inf_and_nan_propagates(tmp_path):        # Review Focus 4
    # set DGS10 to "." on one trading day in the sample files before building
    assert not np.isinf(p.select_dtypes("number")).any().any()
    assert np.isnan(p.loc[day, "DGS10_chg"]) and np.isnan(p.loc[next_day, "DGS10_chg"])
```

- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement. **Step 4:** Run; pass.
- [ ] **Step 5:** Commit: `feat: merged daily panel, returns, rate regimes`.

### Task 6: Question 1, volatility and summary

**Files:** Create `cryptomacro/analysis.py`; Test `tests/test_analysis.py`

**Interfaces:**
- Consumes: the panel from Task 5, `config.CHANGE_COLUMN`.
- Produces:

```python
def rolling_correlation(panel, coin: str, series: str, window: int = config.DEFAULT_WINDOW) -> pd.Series
    # corr of f"{coin}_ret" with CHANGE_COLUMN[series]; min_periods = window; name f"{coin} vs {series}"
def correlation_peak(corr: pd.Series) -> tuple[pd.Timestamp | None, float]   # (None, nan) if all NaN
def annualized_volatility(returns, periods_per_year: int) -> float           # np.nanstd(ddof=1) * sqrt(n)
def weekly_comparison(panel, coin: str, series: str) -> pd.DataFrame
    # index ["daily", "weekly"], columns ["correlation", "n"]; weekly uses resample("W-FRI").last() on levels
def summary_table(panel, coin: str, series: str, window: int = config.DEFAULT_WINDOW) -> pd.DataFrame
    # one row per label, column "value": "Days in sample", "Mean daily return (coin)",
    # "Annualized volatility (coin, x sqrt 365)", "Annualized volatility (S&P 500, x sqrt 252)",
    # "Full-sample correlation", "Peak rolling correlation", "Peak date", "Latest rolling correlation"
```

- [ ] **Step 1: Write failing tests**

```python
def test_rolling_correlation_perfect():
    # panel where BTC_ret == 2 * SP500_ret for 200 rows
    c = rolling_correlation(p, "BTC", "SP500", 90)
    assert c.iloc[:89].isna().all() and c.iloc[89:] == pytest.approx(1.0)

def test_correlation_peak_finds_max_and_date(): ...series with a known max at a known date
def test_window_longer_than_data_gives_nan_not_error():     # Review Focus 5
    c = rolling_correlation(p.iloc[:50], "BTC", "SP500", 90)
    assert c.isna().all() and correlation_peak(c)[0] is None
def test_annualized_volatility_scaling():
    r = np.array([0.01, -0.01, 0.02, -0.02])
    assert annualized_volatility(r, 365) == pytest.approx(np.std(r, ddof=1) * np.sqrt(365))
def test_weekly_comparison_shape(tmp_path): ...sample panel; index == ["daily","weekly"]; weekly n < daily n
def test_summary_table_on_sample(tmp_path):
    t = summary_table(p, "BTC", "SP500", 90)
    assert t.loc["Full-sample correlation", "value"] > 0.3 and t.loc["Days in sample", "value"] == len(p)
```

- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement. **Step 4:** Run; pass.
- [ ] **Step 5:** Commit: `feat: rolling correlation, volatility, summary table`.

### Task 7: Questions 2 and 4

**Files:** Modify `cryptomacro/analysis.py`; Test `tests/test_analysis.py`

**Interfaces — Produces:**

```python
def rate_beta_by_regime(panel, coin: str) -> pd.DataFrame
    # index: "cutting", "holding", "raising", "all"; columns: alpha, beta, corr, n
    # fit f"{coin}_ret" = alpha + beta * DGS10_chg with np.linalg.lstsq on rows where both are present;
    # a group with n < 3 gets NaN for alpha, beta, corr
def zscore(s: pd.Series) -> pd.Series                         # (s - mean) / std(ddof=1)
def hedge_correlation_matrix(panel, coin: str = "BTC") -> pd.DataFrame
    # 3x3 from np.corrcoef over rows where all three are present;
    # labels [coin, "Breakeven inflation", "Dollar index"] from f"{coin}_ret", T10YIE_chg, DTWEXBGS_ret
def hedge_correlation_by_year(panel, coin: str = "BTC") -> pd.DataFrame
    # groupby(index.year); columns: corr_inflation, corr_dollar, n
```

- [ ] **Step 1: Write failing tests**

```python
def test_rate_beta_recovers_known_slope():
    # BTC_ret = 0.001 - 0.5 * DGS10_chg in "raising", = 0.002 + 0.25 * DGS10_chg in "cutting", 100 rows each
    b = rate_beta_by_regime(p, "BTC")
    assert b.loc["raising", "beta"] == pytest.approx(-0.5) and b.loc["raising", "alpha"] == pytest.approx(0.001)
    assert b.loc["cutting", "beta"] == pytest.approx(0.25) and b.loc["raising", "n"] == 100
def test_rate_beta_small_group_is_nan():                     # Review Focus 5
    # "holding" has 2 rows -> beta NaN, n == 2, no exception
def test_rate_beta_ignores_nan_rows(): ...one NaN in DGS10_chg lowers n by 1
def test_zscore(): assert zscore(pd.Series([1., 2., 3.])).tolist() == pytest.approx([-1, 0, 1])
def test_hedge_matrix_signs():
    # BTC_ret = T10YIE_chg + noise, and = -DTWEXBGS_ret + noise
    m = hedge_correlation_matrix(p)
    assert m.loc["BTC", "Breakeven inflation"] > 0.5 and m.loc["BTC", "Dollar index"] < -0.5
    assert np.allclose(np.diag(m), 1.0)
def test_hedge_by_year_index(tmp_path): ...sample panel -> index [2022, 2023]
```

- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement. **Step 4:** Run; pass.
- [ ] **Step 5:** Commit: `feat: rate sensitivity by regime, hedge correlations`.

### Task 8: Question 3 — CPI event study

**Files:** Modify `cryptomacro/analysis.py`; Test `tests/test_event_study.py`

**Interfaces — Produces:**

```python
def release_times_utc(dates: pd.DatetimeIndex) -> pd.DatetimeIndex
    # each date at 08:30 America/New_York, converted to UTC
def hourly_abs_returns(hourly: pd.DataFrame) -> pd.Series
    # reindex to a complete hourly range first, then |log return|; a return touching a missing candle is NaN
def event_matrix(abs_ret: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame
    # rows = dates, columns = relative hours -6..12; hour 0 is the candle containing 08:30 Eastern,
    # found with pd.merge_asof(direction="backward") on the release time; values looked up by timestamp
def daily_event_comparison(hourly: pd.DataFrame, release_dates: pd.DatetimeIndex) -> pd.DataFrame
    # index [False, True] named "cpi_day"; columns realized_vol_mean, realized_vol_std, volume_mean, n_days
    # a "day" is the Eastern calendar date; realized vol = std of that day's hourly log returns; weekdays only
def event_window_profile(hourly: pd.DataFrame, release_dates: pd.DatetimeIndex,
                         draws: int = config.BOOTSTRAP_DRAWS, seed: int = config.SEED) -> pd.DataFrame
    # index rel_hour -6..12; columns cpi_mean, base_mean, base_low, base_high
def reaction_hours(profile: pd.DataFrame) -> int | None
    # None if cpi_mean at hour 0 is not above base_high; otherwise the first hour h >= 1 where
    # cpi_mean <= base_high (the reaction lasted h hours); EVENT_HOURS_AFTER + 1 if it never returns
```

Bootstrap: ordinary days are weekdays in the data's range that are not release dates. Each draw picks `len(release_dates)` ordinary days with replacement using `np.random.default_rng(seed)` and takes the column means of their event-matrix rows (`np.nanmean`). `base_low`/`base_high` are `np.percentile(..., [2.5, 97.5], axis=0)`.

- [ ] **Step 1: Write failing tests**

```python
def test_release_times_handle_daylight_saving():
    t = release_times_utc(pd.DatetimeIndex(["2024-01-11", "2024-07-11"]))
    assert t[0] == pd.Timestamp("2024-01-11 13:30", tz="UTC") and t[1] == pd.Timestamp("2024-07-11 12:30", tz="UTC")

def test_event_matrix_hour_zero_is_candle_containing_release():
    # hourly abs returns equal to the UTC hour number; winter date -> hour 0 value 13, hour -6 value 7, hour 12 value 1
def test_event_matrix_missing_candle_is_nan_not_shifted():    # Review Focus 2
    # drop the 15:00 UTC candle on a winter date -> column 2 and column 3 are NaN (both returns touch it),
    # column 1 is still 14 and column 4 is still 17
def test_profile_detects_three_hour_reaction(tmp_path):
    write_sample_raw(tmp_path); h = load_candles("BTC-USD", 3600, tmp_path); d = load_cpi_release_dates(tmp_path)
    prof = event_window_profile(h, d, draws=500)
    assert list(prof.index) == list(range(-6, 13))
    assert prof.loc[0, "cpi_mean"] > prof.loc[0, "base_high"] and prof.loc[-3, "cpi_mean"] < prof.loc[-3, "base_high"]
    assert reaction_hours(prof) == 3
def test_profile_is_reproducible(tmp_path): ...two calls with the same seed are equal; a different seed changes base_high
def test_reaction_hours_none_when_no_reaction(): ...cpi_mean below base_high everywhere -> None
def test_daily_event_comparison(tmp_path):
    c = daily_event_comparison(h, d)
    assert list(c.index) == [False, True] and c.loc[True, "realized_vol_mean"] > c.loc[False, "realized_vol_mean"]
    assert c.loc[True, "n_days"] == len(d)
```

- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement. **Step 4:** Run; pass.
- [ ] **Step 5:** Commit: `feat: CPI event study with bootstrap baseline`.

### Task 9: Plots, dashboard, notebook, README, package

**Files:** Create `cryptomacro/plots.py`, `cryptomacro/dashboard.py`, `tools/build_notebook.py`, `crypto_macro.ipynb`, `README.md`; Test `tests/test_dashboard.py`

**Interfaces:**
- Consumes: everything above.
- Produces:

```python
# plots.py — each returns a matplotlib Figure and never calls plt.show()
def plot_rolling_correlation(corr, peak, window) -> Figure     # line, zero line, peak marked and labelled
def plot_event_comparison(comparison) -> Figure                # two bar panels: realized vol, volume
def plot_event_profile(profile, reaction) -> Figure            # cpi_mean line, shaded base band, vertical line at 0
def plot_regime_betas(betas) -> Figure                         # bar per regime, n in labels
def plot_hedge(panel, coin, window) -> Figure                  # rolling corr vs breakeven and vs dollar
def plot_zscores(panel, coin) -> Figure                        # z-scored levels of coin, T10YIE, DTWEXBGS

# dashboard.py
def load_all(raw_dir=config.RAW_DIR) -> dict     # {"panel", "hourly": {"BTC": df, "ETH": df}, "cpi_dates"}
def render(data, coin: str, series: str, window: int) -> dict   # {"corr_fig", "event_fig", "summary"}
def build_dashboard(data) -> ipywidgets.VBox
```

- `build_dashboard`: a `Dropdown` for coin (`BTC`, `ETH`), a `Dropdown` for series (labels from `FRED_SERIES`, values the IDs), an `IntSlider` for window (30–180, step 10, default 90, `continuous_update=False`), and an `Output`. Any change clears the output and displays the three results from `render`. Figures are closed after display.
- Notebook (generated by `tools/build_notebook.py` with `nbformat`, then committed): title and one-paragraph intro; a load cell that catches `DataMissingError` and prints its message; the dashboard; then four sections, one per question, each with a two-to-three-sentence plain-language note on what is computed, the table, and the plot. No numbers typed into the text — every figure shown comes from a cell output.
- README: what the project is, install, how to get and set the FRED key, `python -m cryptomacro.acquire` (expected time: about 5 minutes; safe to re-run), `jupyter lab crypto_macro.ipynb`, `pytest`, file map, the weekend-return and regime-threshold choices, data-use note (S&P 500 values not redistributed).

- [ ] **Step 1: Write failing tests**

```python
matplotlib.use("Agg")
def test_render_returns_figures_and_table(tmp_path):
    write_sample_raw(tmp_path); data = load_all(tmp_path)
    out = render(data, "ETH", "DGS10", 60)
    assert isinstance(out["corr_fig"], Figure) and isinstance(out["event_fig"], Figure)
    assert "Full-sample correlation" in out["summary"].index
def test_render_changes_with_inputs(tmp_path):
    a = render(data, "BTC", "SP500", 90)["summary"]; b = render(data, "BTC", "DTWEXBGS", 90)["summary"]
    assert a.loc["Full-sample correlation", "value"] != b.loc["Full-sample correlation", "value"]
def test_build_dashboard_has_three_controls(tmp_path):
    box = build_dashboard(data); controls = box.children[0].children
    assert [type(c).__name__ for c in controls] == ["Dropdown", "Dropdown", "IntSlider"]
    assert controls[2].min == 30 and controls[2].max == 180 and controls[2].value == 90
def test_load_all_missing_data_message(tmp_path):            # Review Focus 3
    with pytest.raises(DataMissingError, match="python -m cryptomacro.acquire"): load_all(tmp_path)
def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):  # Review Focus 3
    monkeypatch.chdir(tmp_path); import cryptomacro.config as c
    assert c.RAW_DIR.is_absolute() and c.RAW_DIR.parts[-2:] == ("data", "raw")
def test_every_plot_function_returns_a_figure(tmp_path): ...call all six on sample data
```

- [ ] **Step 2:** Run; FAIL. **Step 3:** Implement plots, dashboard, the notebook builder; run the builder; write the README. **Step 4:** Run `pytest -v`. Expected: every test in the suite passes.
- [ ] **Step 5: Notebook smoke test.** Write sample data into `data/raw/`, run `jupyter nbconvert --to notebook --execute crypto_macro.ipynb --output /tmp/executed.ipynb`. Expected: exit 0 and no cell with an error output. Open two of the rendered figures as images and check they are readable. Then delete `data/raw/` and re-execute: the load cell prints the "Run: python -m cryptomacro.acquire" message. Clear outputs in the committed notebook.
- [ ] **Step 6:** Check that only `acquire.py` imports requests: `grep -rn "import requests" cryptomacro/` shows one file. Check `git status` shows nothing under `data/`.
- [ ] **Step 7:** Commit: `feat: plots, dashboard, notebook, README`. Zip the project without `.git`, `.venv`, `data/` and caches as `crypto-macro.zip`.
