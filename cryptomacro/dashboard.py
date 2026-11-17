"""Load the saved data once, compute the three dashboard outputs, and wire them to ipywidgets controls."""
import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display

from cryptomacro import analysis, config, plots
from cryptomacro.clean import DataMissingError, build_daily_panel, load_candles, load_cpi_release_dates


def load_all(raw_dir=config.RAW_DIR) -> dict:
    """Everything the notebook needs. Raises DataMissingError (with the fix in its message) if data is absent."""
    # acquire.main writes this file only after every download succeeded, so a partial pull is refused.
    if not (raw_dir / config.PULL_COMPLETE_FILE).exists():
        raise DataMissingError("The data pull has not finished. Run: python -m cryptomacro.acquire "
                               "(it resumes where it stopped)")
    hourly = {coin: load_candles(product, config.GRANULARITIES["hourly"], raw_dir)
              for coin, product in config.PRODUCTS.items()}
    # Daily candles for every calendar day, used for the coin's volatility in the summary table.
    daily = {coin: load_candles(product, config.GRANULARITIES["daily"], raw_dir)
             for coin, product in config.PRODUCTS.items()}
    return {"panel": build_daily_panel(raw_dir), "hourly": hourly, "daily": daily,
            "cpi_dates": load_cpi_release_dates(raw_dir)}


def render(data: dict, coin: str, series: str, window: int) -> dict:
    """The three dashboard outputs for one choice of coin, series and window."""
    panel = data["panel"]
    corr = analysis.rolling_correlation(panel, coin, series, window)
    peak = analysis.correlation_peak(corr)
    comparison = analysis.daily_event_comparison(data["hourly"][coin], data["cpi_dates"])
    return {"corr_fig": plots.plot_rolling_correlation(corr, peak, window),
            "event_fig": plots.plot_event_comparison(comparison),
            "summary": analysis.summary_table(panel, coin, series, window,
                                              coin_daily_close=data["daily"][coin]["close"])}


def build_dashboard(data: dict) -> widgets.VBox:
    """Two dropdowns and a slider above an output area that is redrawn whenever a control changes."""
    coin = widgets.Dropdown(options=list(config.PRODUCTS), value="BTC", description="Coin:")
    series = widgets.Dropdown(options=[(name, series_id) for series_id, name in config.FRED_SERIES.items()],
                              value="SP500", description="Series:")
    window = widgets.IntSlider(value=config.DEFAULT_WINDOW, min=config.WINDOW_MIN, max=config.WINDOW_MAX,
                               step=10, description="Window (days):", continuous_update=False)
    output = widgets.Output()

    def redraw(_change=None):
        result = render(data, coin.value, series.value, window.value)
        output.clear_output(wait=True)
        with output:
            display(result["corr_fig"])
            display(result["event_fig"])
            display(result["summary"])
        plt.close(result["corr_fig"])
        plt.close(result["event_fig"])

    for control in (coin, series, window):
        control.observe(redraw, names="value")
    redraw()  # show the default choice straight away
    return widgets.VBox([widgets.HBox([coin, series, window]), output])
