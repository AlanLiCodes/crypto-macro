import matplotlib
matplotlib.use("Agg")  # no window; must be set before pyplot is imported

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from cryptomacro import analysis, config
from cryptomacro.clean import DataMissingError
from cryptomacro.dashboard import build_dashboard, load_all, render
from cryptomacro.plots import (plot_event_comparison, plot_event_profile, plot_hedge,
                               plot_regime_betas, plot_rolling_correlation, plot_zscores)
from tests.sample_data import write_sample_raw


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    raw = tmp_path_factory.mktemp("raw")
    write_sample_raw(raw)
    return load_all(raw)


def test_load_all_has_panel_hourly_daily_and_cpi_dates(data):
    assert set(data) == {"panel", "hourly", "daily", "cpi_dates"}
    assert set(data["hourly"]) == {"BTC", "ETH"} and set(data["daily"]) == {"BTC", "ETH"}
    # Daily candles cover every calendar day, weekends included.
    assert (data["daily"]["BTC"].index.day_of_week >= 5).any()
    assert "BTC_ret" in data["panel"].columns and len(data["cpi_dates"]) > 0


def test_render_returns_figures_and_table(data):
    out = render(data, "ETH", "DGS10", 60)
    assert isinstance(out["corr_fig"], Figure) and isinstance(out["event_fig"], Figure)
    assert "Full-sample correlation" in out["summary"].index
    plt.close("all")


def test_render_changes_with_inputs(data):
    a = render(data, "BTC", "SP500", 90)["summary"]
    b = render(data, "BTC", "DTWEXBGS", 90)["summary"]
    assert a.loc["Full-sample correlation", "value"] != b.loc["Full-sample correlation", "value"]
    plt.close("all")


def test_render_uses_the_chosen_coins_daily_candles_for_volatility(data):
    summary = render(data, "ETH", "SP500", 90)["summary"]
    expected = analysis.annualized_volatility(
        analysis.calendar_day_log_returns(data["daily"]["ETH"]["close"]), 365)
    assert summary.loc["Annualized volatility (coin, x sqrt 365)", "value"] == expected
    plt.close("all")


def test_render_uses_the_chosen_coin_for_the_event_figure(data):
    out = render(data, "ETH", "SP500", 90)
    expected = analysis.daily_event_comparison(data["hourly"]["ETH"], data["cpi_dates"])
    bar_heights = [bar.get_height() for ax in out["event_fig"].axes for bar in ax.patches]
    assert expected["realized_vol_mean"].iloc[0] in bar_heights
    plt.close("all")


def test_build_dashboard_has_three_controls(data):
    box = build_dashboard(data)
    controls = box.children[0].children
    assert [type(c).__name__ for c in controls] == ["Dropdown", "Dropdown", "IntSlider"]
    assert controls[0].options == ("BTC", "ETH")
    assert dict((label, value) for label, value in controls[1].options) == {
        name: series_id for series_id, name in config.FRED_SERIES.items()}
    assert controls[2].min == 30 and controls[2].max == 180 and controls[2].step == 10
    assert controls[2].value == 90 and controls[2].continuous_update is False
    assert type(box.children[1]).__name__ == "Output"
    plt.close("all")


def test_dashboard_redraws_on_change_and_closes_figures(data, monkeypatch):
    shown = []
    # Outside Jupyter an Output widget captures nothing, so record what would be displayed.
    monkeypatch.setattr("cryptomacro.dashboard.display", shown.append)
    plt.close("all")
    box = build_dashboard(data)
    assert len(shown) == 3  # corr figure, event figure, summary table
    controls = box.children[0].children
    controls[0].value = "ETH"  # a change redraws
    assert len(shown) == 6
    assert [type(item).__name__ for item in shown[3:]] == ["Figure", "Figure", "DataFrame"]
    assert plt.get_fignums() == []  # figures are closed after display


def test_load_all_missing_data_message(tmp_path):
    with pytest.raises(DataMissingError, match="python -m cryptomacro.acquire"):
        load_all(tmp_path)


def test_load_all_refuses_a_pull_without_the_completion_marker(tmp_path):
    write_sample_raw(tmp_path)
    (tmp_path / "pull_complete.json").unlink()  # as if acquire stopped part-way
    with pytest.raises(DataMissingError) as excinfo:
        load_all(tmp_path)
    assert str(excinfo.value) == ("The data pull has not finished. Run: python -m cryptomacro.acquire "
                                  "(it resumes where it stopped)")


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert config.RAW_DIR.is_absolute() and config.RAW_DIR.parts[-2:] == ("data", "raw")


def test_every_plot_function_returns_a_figure(data):
    panel, btc = data["panel"], data["hourly"]["BTC"]
    corr = analysis.rolling_correlation(panel, "BTC", "SP500", 60)
    peak = analysis.correlation_peak(corr)
    comparison = analysis.daily_event_comparison(btc, data["cpi_dates"])
    profile = analysis.event_window_profile(btc, data["cpi_dates"], draws=50)
    figures = [plot_rolling_correlation(corr, peak, 60),
               plot_event_comparison(comparison),
               plot_event_profile(profile, analysis.reaction_hours(profile)),
               plot_regime_betas(analysis.rate_beta_by_regime(panel, "BTC")),
               plot_hedge(panel, "BTC", 60),
               plot_zscores(panel, "BTC")]
    assert all(isinstance(fig, Figure) for fig in figures)
    assert all(len(fig.axes) >= 1 for fig in figures)
    plt.close("all")


def flat_profile(cpi_mean):
    hours = list(range(-config.EVENT_HOURS_BEFORE, config.EVENT_HOURS_AFTER + 1))
    return pd.DataFrame({"cpi_mean": cpi_mean, "base_mean": 2.0, "base_low": 1.5, "base_high": 3.0},
                        index=hours)


def test_event_profile_title_when_the_reaction_never_returns_to_the_band():
    profile = flat_profile(5.0)  # above the band at every hour
    reaction = analysis.reaction_hours(profile)
    assert reaction == config.EVENT_HOURS_AFTER + 1
    title = plot_event_profile(profile, reaction).axes[0].get_title()
    assert "reaction lasts beyond +12 h" in title
    assert "13" not in title
    plt.close("all")


def test_event_profile_title_with_a_measured_reaction_and_none():
    assert "reaction lasts 3 h" in plot_event_profile(flat_profile(1.0), 3).axes[0].get_title()
    assert "no reaction above the band" in plot_event_profile(flat_profile(1.0), None).axes[0].get_title()
    plt.close("all")
