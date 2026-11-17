"""Charts for the notebook. Each function builds one matplotlib Figure and returns it.

Nothing here calls plt.show(); the caller decides when to display and close the figure.
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from cryptomacro import analysis, config


def _tidy_dates(ax) -> None:
    """Short, non-overlapping date labels on the x-axis."""
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def plot_rolling_correlation(corr: pd.Series, peak: tuple, window: int) -> Figure:
    """Rolling correlation as a line, a zero line, and the highest point marked and labelled."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(corr.index, corr.values, color="tab:blue", label=f"{window}-day rolling correlation")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.margins(y=0.2)  # headroom so the peak label stays inside the chart
    peak_date, peak_value = peak
    if peak_date is not None:
        ax.scatter([peak_date], [peak_value], color="tab:red", zorder=3, label="Peak")
        ax.annotate(f"{peak_value:.2f} on {peak_date:%Y-%m-%d}", (peak_date, peak_value),
                    textcoords="offset points", xytext=(8, 10), color="tab:red")
    ax.set_title(f"Rolling correlation: {corr.name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation (-1 to 1)")
    ax.legend(loc="lower left")
    _tidy_dates(ax)
    fig.tight_layout()
    return fig


def plot_event_comparison(comparison: pd.DataFrame) -> Figure:
    """Two bar panels, CPI days versus other weekdays: realized volatility and trading volume."""
    labels = ["Other weekdays", "CPI days"]  # table rows are cpi_day False, True
    colors = ["tab:grey", "tab:orange"]
    fig, (ax_vol, ax_volume) = plt.subplots(1, 2, figsize=(9, 4))
    ax_vol.bar(labels, comparison["realized_vol_mean"], yerr=comparison["realized_vol_std"],
               color=colors, capsize=4)
    ax_vol.set_title("Realized volatility")
    ax_vol.set_ylabel("Std. dev. of hourly log returns")
    ax_volume.bar(labels, comparison["volume_mean"], color=colors)
    ax_volume.set_title("Trading volume")
    ax_volume.set_ylabel("Coins traded per day")
    fig.suptitle("CPI days versus other weekdays (error bars: one standard deviation across days)")
    fig.tight_layout()
    return fig


def plot_event_profile(profile: pd.DataFrame, reaction) -> Figure:
    """Mean absolute hourly return around the release, with the ordinary-day baseline band."""
    fig, ax = plt.subplots(figsize=(9, 4))
    hours = profile.index
    ax.fill_between(hours, profile["base_low"], profile["base_high"], color="tab:grey", alpha=0.3,
                    label="Ordinary days (95% band)")
    ax.plot(hours, profile["cpi_mean"], color="tab:orange", marker="o", label="CPI days (mean)")
    ax.axvline(0, color="black", linestyle="--", linewidth=1, label="Release hour (08:30 Eastern)")
    if reaction is None:
        note = "no reaction above the band"
    elif reaction > config.EVENT_HOURS_AFTER:
        # reaction_hours returns EVENT_HOURS_AFTER + 1 when the line never comes back inside the window.
        note = f"reaction lasts beyond +{config.EVENT_HOURS_AFTER} h"
    else:
        note = f"reaction lasts {reaction} h"
    ax.set_title(f"Mean absolute hourly return around CPI releases ({note})")
    ax.set_xticks(list(hours))  # whole hours only
    ax.set_xlabel("Hours relative to the release hour")
    ax.set_ylabel("Mean |hourly log return|")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_regime_betas(betas: pd.DataFrame) -> Figure:
    """One bar per rate regime (and "all"): the slope on the 10-year yield change, with n in the label."""
    labels = [f"{regime}\n(n={int(row.n)})" for regime, row in betas.iterrows()]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(labels, betas["beta"], color=["tab:green", "tab:blue", "tab:red", "tab:grey"][:len(betas)])
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Sensitivity to the 10-year yield by Fed regime")
    ax.set_xlabel("Fed regime (number of trading days)")
    ax.set_ylabel("Beta (return per 1 pt yield change)")
    fig.tight_layout()
    return fig


def plot_hedge(panel: pd.DataFrame, coin: str, window: int) -> Figure:
    """Rolling correlation of the coin with breakeven inflation and with the dollar index."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for series_id, color in [("T10YIE", "tab:green"), ("DTWEXBGS", "tab:purple")]:
        corr = analysis.rolling_correlation(panel, coin, series_id, window)
        ax.plot(corr.index, corr.values, color=color, label=f"vs {config.FRED_SERIES[series_id]}")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title(f"{coin}: {window}-day rolling correlation with inflation expectations and the dollar")
    ax.set_xlabel("Date")
    ax.set_ylabel("Correlation (-1 to 1)")
    ax.legend(loc="lower left")
    _tidy_dates(ax)
    fig.tight_layout()
    return fig


def plot_zscores(panel: pd.DataFrame, coin: str) -> Figure:
    """Levels of the coin, breakeven inflation and the dollar index, each standardized to compare shapes."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for column, label in [(coin, coin), ("T10YIE", config.FRED_SERIES["T10YIE"]),
                          ("DTWEXBGS", config.FRED_SERIES["DTWEXBGS"])]:
        z = analysis.zscore(panel[column].dropna())
        ax.plot(z.index, z.values, label=label)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_title("Levels compared on one scale (z-scores)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Standard deviations from the series mean")
    ax.legend()
    _tidy_dates(ax)
    fig.tight_layout()
    return fig
