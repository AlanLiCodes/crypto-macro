# Crypto and the Macro Economy — build design

Date: 2026-11-10
For: Alan Li and Yi-Sheng Chao, DSCI 510 course project
Source: the submitted project proposal ("DSCI 510 Project Proposal — Crypto and the Macro Economy")

## Intent

Build the whole project the proposal describes: collect Bitcoin and Ethereum prices from Coinbase and economic series from FRED, answer four questions about how crypto moves with the wider economy, and present the results in a Jupyter notebook with interactive widgets.

The why: it is a graded course project. It has to do what the proposal promised, run on a grader's or teammate's machine, and be code both team members can read and explain.

## Goals

- `acquire.py` fetches everything through the two official APIs and saves raw JSON to disk; the analysis never calls the APIs.
- Every technique the proposal names is really used: a generator for paging, comprehensions for flattening, retry with backoff, caching and resume, pandas cleaning/aligning/groupby/resample/merge_asof/pd.cut, NumPy log returns, lstsq, bootstrap with default_rng and percentile, corrcoef.
- Each of the four questions has one function that returns its result as a DataFrame, and one plot.
- The notebook lets a user pick a coin, an economic series and a window length, and redraws the rolling correlation, the event-day volatility comparison and a summary table.
- All results are computed from the collected data. Nothing is hard-coded.

## Non-goals

- No web scraping, no other data sources, no coins beyond BTC and ETH.
- No forecasting, trading signals or causal claims; the analysis is descriptive.
- No final report, slides or demo script in this build (they can follow as a separate request).

## Anti-goals

- An API key in the code or the repository.
- Raw S&P 500 values in the repository (copyright; `data/` is git-ignored).
- Analysis code that silently re-downloads data.
- Clever code the team cannot explain in a demo. Plain functions, short modules, comments where a choice is not obvious.

## Constraints

- The build environment cannot reach either API. All code is tested against sample responses shaped like the documented ones. The real pull runs on Alan's machine with his own FRED key, and the first real run may surface small differences that need a fix.
- Coinbase returns at most 300 candles per request, newest first, with a public limit of 10 requests per second.
- FRED needs a free key, read from the `FRED_API_KEY` environment variable. FRED marks missing values with ".".
- FRED gives CPI release dates only. The release time is 8:30 AM Eastern, which has to be converted to UTC with daylight saving handled.
- Period: 2020-01-01 to 2026-09-30, as in the proposal.

## Approaches considered

1. **Modules plus a thin notebook (chosen by Alan).** Code lives in a small package; the notebook imports it and wires up widgets. Testable without the network, and each file maps to a responsibility in the proposal. Costs a few more files to submit.
2. **Everything in one notebook.** Simplest to submit, but cannot be tested from the build environment and drifts from the proposal, which names `acquire.py` as a module.
3. **Modules, one question at a time end to end.** Gets something runnable on real data soonest, but shared pieces get reworked as later questions add needs.

Chosen: option 1, with the build order borrowed from option 3 — shared acquisition and cleaning first, then questions 1, 2, 4, 3 (question 3 is the only one needing hourly data and time zones).

## Layout

```
crypto-macro/
  README.md               how to install, set the key, pull data, open the notebook
  requirements.txt        requests, pandas, numpy, matplotlib, ipywidgets, jupyter, pytest
  .gitignore              data/, .env, caches
  cryptomacro/
    config.py             period, product IDs, FRED series IDs, paths
    acquire.py            all network access and raw-JSON caching
    clean.py              raw JSON -> tidy DataFrames, merged daily panel, regime labels
    analysis.py           one function per question, plus summary table
    plots.py              one matplotlib figure per result
  crypto_macro.ipynb      widgets + the four question sections
  tests/                  pytest, with small sample responses; no network
  data/raw/<source>/      created by acquire.py (not committed)
```

## Acquisition (`acquire.py`)

- `get_json(url, params)`: one shared function. Sets a timeout, calls `raise_for_status()`, retries with exponential backoff on HTTP 429 and 5xx (up to 5 tries), and sleeps between calls (0.25 s for Coinbase, 1 s for FRED).
- `candle_windows(start, end, granularity)`: a generator yielding start/end pairs of at most 300 candles (300 days daily, 12.5 days hourly).
- `fetch_candles(product, granularity)`: one request per window, saved as `data/raw/coinbase/<product>_<granularity>_<start>.json`. Windows already on disk are skipped, so an interrupted run resumes. Empty windows are saved as empty lists and tolerated.
- `fetch_fred_series(series_id)`: one call covering the period, saved as `data/raw/fred/<series_id>.json`.
- `fetch_cpi_release_dates()`: one call to `release/dates?release_id=10`, saved as `data/raw/fred/cpi_release_dates.json`.
- A missing or rejected FRED key produces one clear error message, not a traceback.
- Run with `python -m cryptomacro.acquire`. It prints progress and a final count of files fetched and skipped. About 410 Coinbase requests and 7 FRED requests in total.

## Cleaning and merging (`clean.py`)

- Candles: flatten `[time, low, high, open, close, volume]` arrays with a comprehension, convert Unix time to a UTC DatetimeIndex, sort oldest first, drop duplicate timestamps from overlapping windows.
- FRED: convert date strings to datetimes and "." to NaN.
- Daily panel: crypto daily closes joined to the six FRED series on date, filtered to US trading days (days the S&P 500 has a value).
- Returns: log returns with NumPy. Crypto returns in the panel run from one trading day's close to the next, so weekend moves fold into Monday (builder's call; keeps crypto and stock returns covering the same span).
- Changes: `diff` for yields and breakeven inflation, `pct_change` for the dollar index.
- Rate regime: change in the fed funds rate over the previous 90 days, cut with `pd.cut` into cutting (≤ −0.125 points), holding, raising (≥ +0.125 points) (builder's call on the thresholds; they sit in `config.py`).

## Analysis (`analysis.py`)

1. **Tech-stock question.** Rolling correlation (default 90 trading days) between a coin's returns and S&P 500 or Nasdaq returns; reports the peak value and its date.
2. **Interest-rate question.** For each regime, a least-squares fit with `np.linalg.lstsq` of coin returns on 10-year yield changes: intercept, slope (beta), correlation, number of days.
3. **CPI-day question.**
   - Daily view: realized volatility (std of that day's hourly returns) and volume, grouped by CPI-day flag.
   - Hourly view: mean absolute hourly return for hours −6 to +12 around the release. Hour 0 is the hourly candle that contains 8:30 AM Eastern. `merge_asof` matches each release to its candle.
   - Baseline: draw the same number of ordinary weekdays at random, 2,000 times, with a seeded `np.random.default_rng`; the 95% band comes from `np.percentile`.
   - Reaction length: the first hour after the release where the CPI-day average falls back inside the band.
4. **Hedge question.** Correlation matrix (`np.corrcoef`) of Bitcoin returns, breakeven-inflation changes and dollar-index returns; rolling correlations over time; the same by year with `groupby`; z-scored levels for the plot.

Also: annualized volatility (daily std × √365 for crypto, × √252 for stocks), a `describe`-based summary table, and a daily-versus-weekly comparison using `resample`.

## Notebook

```
[ Coin: BTC v ]  [ Series: S&P 500 v ]  [ Window: ---o--- 90 days ]

  Rolling correlation: BTC vs S&P 500 (90-day)        <- line chart, peak marked
  CPI days vs ordinary days: volatility and volume    <- bar chart
  Summary table                                       <- returns, volatility, correlation, n

Then four sections, one per question, each with a short
explanation, its table and its plot.
```

Controls are ipywidgets dropdowns and a slider (window 30 to 180 days); changing any of them redraws the three outputs. The notebook reads only from `data/raw/`; if the data is missing it says to run the acquisition first.

## Testing

pytest, no network. A fake session feeds `get_json` canned responses to check retry, backoff, paging, caching and resume, the newest-first ordering, the "." placeholder and the missing-key message. Analysis functions are checked on small synthetic data with known answers (a series perfectly correlated with another gives 1.0; a known slope is recovered by the regression).

## Left to the builder

- Function signatures and internal helper names.
- Plot styling.
- Exact wording of the README and notebook explanations.

## Assumptions to confirm

- The course does not require a specific folder structure, file names or submission format beyond what the proposal says.
- Delivery is a zip of the project folder sent in the chat (it can instead be written into a folder on Alan's computer if he connects one).
- Python 3.10 or newer.
