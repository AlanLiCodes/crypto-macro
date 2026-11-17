"""Run single notebook cells (as written by tools/build_notebook.py) against small stand-in data."""
import matplotlib
matplotlib.use("Agg")  # no window; must be set before pyplot is imported

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cryptomacro import analysis, config, plots
from cryptomacro.clean import DataMissingError
from tools import build_notebook


def run_cell(code, namespace):
    """Execute one notebook cell's code (no IPython magics in these cells) in the given namespace."""
    exec(code, namespace)
    return namespace


def base_namespace(data):
    shown = []
    return {"analysis": analysis, "config": config, "plots": plots, "plt": plt, "pd": pd,
            "display": shown.append, "data": data, "shown": shown}


def short_panel(rows=50):
    """Too few rows for a 90-day window, so every rolling correlation is NaN."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2022-01-03", periods=rows, name="date")
    columns = {}
    for coin in config.PRODUCTS:
        columns[coin] = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, rows)))
        columns[f"{coin}_ret"] = rng.normal(0, 0.02, rows)
    for series in ["SP500", "NASDAQCOM"]:
        columns[series] = 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, rows)))
        columns[f"{series}_ret"] = rng.normal(0, 0.01, rows)
    return pd.DataFrame(columns, index=index)


def test_q1_plot_cell_handles_no_peak(capsys):
    namespace = base_namespace({"panel": short_panel()})
    run_cell(build_notebook.Q1_TABLE, namespace)
    run_cell(build_notebook.Q1_PLOT, namespace)
    out = capsys.readouterr().out
    assert "BTC vs SP500: no peak \u2014 not enough data" in out
    plt.close("all")


def test_q3_plot_cell_says_beyond_when_the_reaction_never_ends(monkeypatch, capsys):
    hours = list(range(-config.EVENT_HOURS_BEFORE, config.EVENT_HOURS_AFTER + 1))
    profile = pd.DataFrame({"cpi_mean": 5.0, "base_mean": 2.0, "base_low": 1.5, "base_high": 3.0},
                           index=hours)
    monkeypatch.setattr(analysis, "event_window_profile", lambda hourly, dates: profile)
    namespace = base_namespace({"hourly": {"BTC": None}, "cpi_dates": None})
    run_cell(build_notebook.Q3_PLOT, namespace)
    out = capsys.readouterr().out
    assert "lasts beyond +12 h" in out and "13" not in out
    plt.close("all")


def test_load_cell_reports_cpi_dates_used_and_dropped(capsys):
    data = {"cpi_dates": pd.DatetimeIndex(["2020-01-14", "2020-02-13"])}
    namespace = base_namespace(None)
    namespace.update(load_all=lambda: data, DataMissingError=DataMissingError,
                     load_dropped_cpi_dates=lambda: pd.DatetimeIndex(["2020-02-03"]))
    run_cell(build_notebook.LOAD, namespace)
    out = capsys.readouterr().out
    assert "CPI release dates used: 2" in out
    assert "2020-02-03" in out


def test_load_cell_prints_the_missing_data_message(capsys):
    def missing():
        raise DataMissingError("The data pull has not finished. Run: python -m cryptomacro.acquire")
    namespace = base_namespace(None)
    namespace.update(load_all=missing, DataMissingError=DataMissingError,
                     load_dropped_cpi_dates=lambda: pd.DatetimeIndex([]))
    run_cell(build_notebook.LOAD, namespace)
    assert namespace["data"] is None
    assert "Run: python -m cryptomacro.acquire" in capsys.readouterr().out


def test_data_at_a_glance_comes_after_loading_and_before_the_dashboard():
    sources = [cell.source for cell in build_notebook.build().cells]
    load = sources.index(build_notebook.LOAD)
    glance = sources.index(build_notebook.GLANCE)
    dashboard = sources.index(build_notebook.DASHBOARD)
    assert load < glance < dashboard
    assert sources[glance - 1].startswith("## The data at a glance")
    namespace = base_namespace({"panel": pd.DataFrame(
        {column: [0.1, 0.2, 0.3] for column in ["BTC_ret", "ETH_ret", *config.CHANGE_COLUMN.values()]})})
    run_cell(build_notebook.GLANCE, namespace)
    assert namespace["shown"][0].loc["count", "BTC_ret"] == 3
