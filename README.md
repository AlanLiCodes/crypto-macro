# Crypto and the macro environment

A DSCI 510 course project. It asks how Bitcoin and Ethereum move together with US stocks, interest rates,
inflation expectations and the dollar between 2020-01-01 and 2026-09-30, and how they react to CPI
inflation releases. Crypto prices come from the Coinbase Exchange API; macro data and CPI release dates come
from FRED (the St. Louis Fed). The results are presented in one Jupyter notebook, `crypto_macro.ipynb`, with an
interactive dashboard and four sections, one per question:

1. Does crypto move with the stock market (S&P 500, Nasdaq)?
2. Does crypto react to interest rates differently when the Fed is cutting, holding or raising?
3. What happens to crypto around CPI releases (volatility, volume, hour-by-hour reaction)?
4. Is crypto a hedge against inflation expectations or a weaker dollar?

## Requirements

- Python 3.10 or newer
- A free FRED API key (see below)
- An internet connection for the one-time data download

## Install

```
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
.venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

## Get a FRED API key and set it

1. Create a free account at https://fred.stlouisfed.org and request a key under "My Account" -> "API Keys".
2. Set it as an environment variable in the terminal you will use to download the data. The key is read from
   the environment only and is never written to a file.

```
export FRED_API_KEY=your_key_here          # macOS / Linux
$env:FRED_API_KEY = "your_key_here"        # Windows PowerShell
```

Or, instead, create a file named `.env` in the project folder containing one line:

```
FRED_API_KEY=your_key_here
```

The `.env` file is git-ignored. Do not commit it, and remove it before sharing or
submitting the folder as a zip.

## Download the data

```
python -m cryptomacro.acquire
```

This makes about 420 requests and takes roughly 5 minutes (it pauses between calls to respect the APIs'
limits). The six FRED series and the CPI release dates are downloaded first (a few seconds), so a wrong FRED key
is reported straight away; the Coinbase candles follow. It is safe to re-run: files that were already saved are
skipped, so an interrupted download resumes where it stopped. Everything is saved under `data/raw/`.

When every download has succeeded, the script writes `data/raw/pull_complete.json` (the period and the time of
completion, in UTC). The notebook refuses to load without it, so a half-finished download is never analysed by
mistake; just run the command again and it continues.

## Run the notebook

```
jupyter lab crypto_macro.ipynb
```

Open it from the project folder and run all cells. If the data has not been downloaded (or the download has not
finished), the load cell prints `Run: python -m cryptomacro.acquire` and the later cells are skipped. Otherwise it
prints how many CPI release dates are used and lists any that were dropped (see the method notes).

## Run the tests

```
pytest
```

The tests never use the network; they run on small generated data shaped like the API responses.

## File map

| Path | What it is |
| --- | --- |
| `cryptomacro/config.py` | Every tunable number and name (dates, series, thresholds, pauses, seed) |
| `cryptomacro/acquire.py` | Downloads Coinbase candles and FRED data into `data/raw/` (the only file that uses the network) |
| `cryptomacro/clean.py` | Reads the raw files; builds the daily panel (returns, changes, rate regimes) |
| `cryptomacro/analysis.py` | Rolling correlation, summary table, regime betas, hedge correlations, CPI event study |
| `cryptomacro/plots.py` | The charts (each function returns a matplotlib figure) |
| `cryptomacro/dashboard.py` | Loads everything and builds the interactive controls |
| `tools/build_notebook.py` | Regenerates `crypto_macro.ipynb` (outputs cleared) |
| `crypto_macro.ipynb` | The deliverable notebook |
| `tests/` | pytest suite and the sample-data generator |
| `docs/` | Design notes |

## Method choices

- **Weekend moves fold into Monday.** Crypto trades every day but US markets do not. The daily panel keeps
  only days with an S&P 500 value, so a Monday return covers the whole weekend. Nothing is forward-filled.
- **Daily closes are a few hours apart.** Coinbase daily candles close at 00:00 UTC (7-8 PM Eastern), not at the
  4 PM stock close, so a "daily" crypto return and a daily stock return end a few hours apart.
- **Rolling windows allow a few missing days.** On Columbus Day and Veterans Day the stock market is open but the
  bond market is closed, so the Treasury, breakeven and dollar series have no value. A rolling correlation is
  reported when at least 80% of the days in its window have both values; the missing days are left out (never
  treated as zero).
- **Volatility.** The coin's annualized volatility uses its daily candles for every calendar day, times the
  square root of 365; the S&P 500's uses trading-day returns, times the square root of 252.
- **Rate regimes.** Each day is labelled from the 90-day change in the effective fed funds rate (DFF): a change
  of -0.125 percentage points or lower is "cutting", +0.125 or higher is "raising", anything between is
  "holding". DFF is downloaded from 100 days before 2020-01-01, so every day of the period (including the March
  2020 cuts) has a label.
- **CPI release dates.** FRED's release calendar is used with at most one date per calendar month; if a month
  lists more than one, the latest is kept and the notebook lists the dropped dates. After the first real download,
  compare the count the notebook prints (about 80) with the BLS CPI release schedule.
- **CPI hour 0.** Releases are at 8:30 AM Eastern (daylight saving handled). Hour 0 is the hourly candle that
  contains 8:30; the event window runs from 6 hours before to 12 hours after.
- **Baseline for the CPI study.** Ordinary weekdays are resampled with a bootstrap of 2,000 draws and a fixed
  random seed (510), so results are repeatable. The band shown is the 2.5th to 97.5th percentile.

## Data use

The raw data is not committed to this repository (`data/` is git-ignored); each user downloads it with the
acquire script. S&P 500 values come from FRED under the index provider's terms and are not redistributed here.

## Testing status

The code has been tested against sample data shaped like the API responses; the first real data pull is the
first contact with the live APIs.
