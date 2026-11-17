"""Write crypto_macro.ipynb at the project root (outputs cleared).

Run from anywhere:  python tools/build_notebook.py
The notebook cells are short calls into the cryptomacro package; all logic lives there.
"""
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

OUTPUT = Path(__file__).resolve().parent.parent / "crypto_macro.ipynb"  # project root

TITLE = """# Crypto and the macro environment

This notebook looks at how Bitcoin and Ethereum move together with US markets and macroeconomic data
between 2020 and September 2026. Crypto prices come from Coinbase and the macro series from FRED
(the St. Louis Fed). It asks four questions: how closely crypto follows stocks, how it reacts to
interest-rate changes in different Fed regimes, how it behaves around CPI inflation releases, and whether it acts
as a hedge against inflation or a weaker dollar. Everything shown below is computed from the saved
files in `data/raw/`; run `python -m cryptomacro.acquire` once first to download them."""

SETUP = """import sys
sys.path.insert(0, ".")  # the notebook is opened from the project folder, which holds cryptomacro/

import matplotlib.pyplot as plt
import pandas as pd

from cryptomacro import analysis, config, plots
from cryptomacro.clean import DataMissingError, load_dropped_cpi_dates
from cryptomacro.dashboard import build_dashboard, load_all

%matplotlib inline"""

LOAD = """try:
    data = load_all()
except DataMissingError as error:
    print(error)
    data = None

if data is not None:
    # Compare this count (about 80) with the BLS CPI schedule after the first real download.
    print(f"CPI release dates used: {len(data['cpi_dates'])}")
    dropped = load_dropped_cpi_dates()
    if len(dropped):
        print("Dropped (a later date in the same month was kept):",
              ", ".join(f"{day:%Y-%m-%d}" for day in dropped))"""

GLANCE_NOTE = """## The data at a glance

Count, mean, spread and range of each daily change in the panel (returns for prices, differences for rates)."""

GLANCE = """if data is not None:
    display(analysis.describe_changes(data["panel"]))"""

DASHBOARD_NOTE = """## Explore: pick a coin, a series and a window

Choose a coin, one of the six macro series and the rolling-window length. The first chart shows the rolling
correlation between the coin's daily return and the series' daily change, with its highest point marked. The
second compares volatility and volume on CPI release days with other weekdays. The table summarizes the choice;
the coin's annualized volatility uses its daily candles for every calendar day (x sqrt 365), and the S&P 500's uses
trading days (x sqrt 252)."""

DASHBOARD = """if data is not None:
    display(build_dashboard(data))"""

Q1_NOTE = """## Question 1: Does crypto move with the stock market?

For each day we take the coin's log return and the index's log return, then compute their correlation over a
rolling window (90 days by default). A value near 1 means they tend to move the same way, near 0 means no
link, and below 0 means opposite directions. The table also compares daily changes with weekly changes, because
daily data mixes weekends (crypto trades, stocks do not) into Monday's move."""

Q1_TABLE = """PAIRS = [(coin, series) for coin in config.PRODUCTS for series in ["SP500", "NASDAQCOM"]]
if data is not None:
    tables = {f"{coin} vs {series}": analysis.weekly_comparison(data["panel"], coin, series) for coin, series in PAIRS}
    display(pd.concat(tables, names=["pair", "frequency"]))"""

Q1_PLOT = """if data is not None:
    for coin, series in PAIRS:
        corr = analysis.rolling_correlation(data["panel"], coin, series, config.DEFAULT_WINDOW)
        peak_date, peak_value = analysis.correlation_peak(corr)
        if peak_date is None:
            print(f"{coin} vs {series}: no peak \u2014 not enough data")
        else:
            print(f"{coin} vs {series}: peak {peak_value:.2f} on {peak_date:%Y-%m-%d}")
        plots.plot_rolling_correlation(corr, (peak_date, peak_value), config.DEFAULT_WINDOW)
        plt.show()"""

Q2_NOTE = """## Question 2: Does crypto react to interest rates differently when the Fed is cutting, holding or raising?

Each day is labelled by the change in the fed funds rate over the previous 90 days: a fall of at least 0.125
percentage points is "cutting", a rise of at least 0.125 is "raising", anything in between is "holding".
Within each regime we fit a straight line of the coin's daily return on the daily change in the 10-year
Treasury yield. The slope (beta) says how much the return moves per one percentage point change in the yield;
the n column shows how many days each estimate rests on, so read small n with caution."""

Q2_TABLE = """if data is not None:
    betas = analysis.rate_beta_by_regime(data["panel"], "BTC")
    display(betas)"""

Q2_PLOT = """if data is not None:
    plots.plot_regime_betas(betas)
    plt.show()"""

Q3_NOTE = """## Question 3: What happens to crypto around CPI inflation releases?

CPI is published at 8:30 AM Eastern. We use hourly candles and call the candle containing 8:30 "hour 0". The
table compares realized volatility (the spread of hourly returns) and trading volume on CPI days with other
weekdays. The profile plot shows the average absolute hourly return from 6 hours before to 12 hours after the
release; the grey band is where ordinary days land 95% of the time (found by resampling ordinary days 2,000
times), so CPI hours above the band are unusually large moves. The reaction length is the number of hours until
the CPI line is back inside the band."""

Q3_TABLE = """if data is not None:
    comparison = analysis.daily_event_comparison(data["hourly"]["BTC"], data["cpi_dates"])
    display(comparison)
    plots.plot_event_comparison(comparison)
    plt.show()"""

Q3_PLOT = """if data is not None:
    profile = analysis.event_window_profile(data["hourly"]["BTC"], data["cpi_dates"])
    reaction = analysis.reaction_hours(profile)
    plots.plot_event_profile(profile, reaction)
    plt.show()
    if reaction is None:
        print("Hour 0 is not above the ordinary-day band: no CPI reaction detected.")
    elif reaction > config.EVENT_HOURS_AFTER:
        print(f"The CPI reaction lasts beyond +{config.EVENT_HOURS_AFTER} h (still above the band at the end of the window).")
    else:
        print(f"The CPI reaction lasts {reaction} hour(s).")"""

Q4_NOTE = """## Question 4: Is crypto a hedge against inflation or a weaker dollar?

We correlate the coin's daily return with the daily change in 10-year breakeven inflation (the market's
inflation expectation) and with the daily return of the trade-weighted dollar index. A hedge would show a
positive link to inflation expectations and a negative link to the dollar. The tables give the full-sample
correlations and one row per calendar year; the line chart shows how the links drift over time, and the last
chart puts the three levels on one scale (z-scores) to compare their long-run shapes."""

Q4_TABLE = """if data is not None:
    display(analysis.hedge_correlation_matrix(data["panel"], "BTC"))
    display(analysis.hedge_correlation_by_year(data["panel"], "BTC"))"""

Q4_PLOT = """if data is not None:
    plots.plot_hedge(data["panel"], "BTC", config.DEFAULT_WINDOW)
    plt.show()
    plots.plot_zscores(data["panel"], "BTC")
    plt.show()"""


def build() -> nbformat.NotebookNode:
    cells = [new_markdown_cell(TITLE), new_code_cell(SETUP), new_code_cell(LOAD),
             new_markdown_cell(GLANCE_NOTE), new_code_cell(GLANCE),
             new_markdown_cell(DASHBOARD_NOTE), new_code_cell(DASHBOARD),
             new_markdown_cell(Q1_NOTE), new_code_cell(Q1_TABLE), new_code_cell(Q1_PLOT),
             new_markdown_cell(Q2_NOTE), new_code_cell(Q2_TABLE), new_code_cell(Q2_PLOT),
             new_markdown_cell(Q3_NOTE), new_code_cell(Q3_TABLE), new_code_cell(Q3_PLOT),
             new_markdown_cell(Q4_NOTE), new_code_cell(Q4_TABLE), new_code_cell(Q4_PLOT)]
    return new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                                                "name": "python3"}})


if __name__ == "__main__":
    notebook = build()
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Wrote {OUTPUT}")
