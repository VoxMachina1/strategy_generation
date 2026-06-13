"""
Monte Carlo simulation and walk-forward testing for the Composer Signal Pipeline.

Ported from:
  monte_carlo_sim/Monte Carlo walk forward composer working.py

All four core functions are ported unchanged in logic. The only adaptation is that
the pipeline interfaces (run_mc_for_signal / run_mc_for_portfolio) accept decimal-scale
returns (as produced by pct_change()) and convert them to percent-scale before calling
the MC engine, which uses percent-scale internally (1.5 = 1.5% daily return).

Public API
----------
run_monte_carlo_simulation()  — bootstrap-resample simulation, positive/negative split
analyze_drawdowns()           — drawdown period detection and visualisation
plot_drawdown_distributions() — side-by-side histogram of simulated vs actual DD
run_walk_forward_test()       — split history into train/test, simulate, compare
run_mc_for_signal()           — pipeline entry: signal-masked returns → walk-forward MC
run_mc_for_portfolio()        — pipeline entry: pre-computed portfolio returns → MC
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats


# ---------------------------------------------------------------------------
# 6.1  Core simulation
# ---------------------------------------------------------------------------

def run_monte_carlo_simulation(
    returns,
    num_simulations: int = 10000,
    simulation_length: int = None,
    annual_periods: int = 252,
) -> dict:
    """
    Run Monte Carlo simulation by bootstrap-resampling from historical returns.

    Sampling is split: positive and negative returns are drawn independently,
    preserving the empirical win-rate while allowing their magnitudes to vary.

    Parameters
    ----------
    returns          : array-like of *percent-scale* daily returns
                       (e.g. 1.5 means +1.5% on that day)
    num_simulations  : number of simulation paths (default 10 000)
    simulation_length: number of trading days to simulate; defaults to len(returns)
    annual_periods   : trading days per year for Sharpe calculation (default 252)

    Returns
    -------
    dict with keys:
        final_returns          : np.ndarray (num_simulations,) — terminal percent return
        paths                  : np.ndarray (num_simulations, simulation_length+1)
        percentiles            : dict {"5","25","50","75","95"} → np.ndarray
        sharpe_ratios          : np.ndarray (num_simulations,)
        max_drawdowns          : np.ndarray (num_simulations,) — percent-scale
        max_drawdown_durations : np.ndarray (num_simulations,) — trading days
        total_drawdown_days    : np.ndarray (num_simulations,)
    """
    if simulation_length is None:
        simulation_length = len(returns)

    returns_array = np.array(returns)

    # Separate positive and negative returns
    positive_returns = returns_array[returns_array > 0]
    negative_returns = returns_array[returns_array <= 0]

    # Calculate probabilities
    prob_positive = len(positive_returns) / len(returns_array)


    # Initialize arrays to store simulation results
    cumulative_returns = np.zeros((num_simulations, simulation_length + 1))
    cumulative_returns[:, 0] = 0  # Start with 0% return

    sharpe_ratios = np.zeros(num_simulations)
    max_drawdowns = np.zeros(num_simulations)
    max_drawdown_durations = np.zeros(num_simulations)
    total_drawdown_days = np.zeros(num_simulations)

    for i in range(num_simulations):
        # Generate random returns by sampling separately from positive and negative
        simulated_returns = np.zeros(simulation_length)
        for j in range(simulation_length):
            if np.random.random() < prob_positive:
                simulated_returns[j] = np.random.choice(positive_returns)
            else:
                simulated_returns[j] = np.random.choice(negative_returns)

        # Calculate cumulative returns
        cum_return = 0.0
        cum_returns = [cum_return]
        peak = 0.0
        max_drawdown = 0.0

        in_drawdown = False
        current_drawdown_duration = 0
        max_dd_duration = 0
        total_dd_days = 0

        for r in simulated_returns:
            r_decimal = r / 100.0
            cum_return = (1 + cum_return / 100) * (1 + r_decimal) * 100 - 100
            cum_returns.append(cum_return)

            if cum_return > peak:
                peak = cum_return
                if in_drawdown:
                    in_drawdown = False
                    max_dd_duration = max(max_dd_duration, current_drawdown_duration)
                    current_drawdown_duration = 0

            drawdown = ((peak - cum_return) / (1 + peak / 100)) if peak > 0 else 0

            if drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                current_drawdown_duration += 1
                total_dd_days += 1

            max_drawdown = max(max_drawdown, drawdown)

        if in_drawdown:
            max_dd_duration = max(max_dd_duration, current_drawdown_duration)

        cumulative_returns[i, :] = cum_returns

        annual_return = cum_return * (annual_periods / simulation_length)
        annual_volatility = np.std(simulated_returns) * np.sqrt(annual_periods)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0

        sharpe_ratios[i] = sharpe_ratio
        max_drawdowns[i] = max_drawdown
        max_drawdown_durations[i] = max_dd_duration
        total_drawdown_days[i] = total_dd_days

    percentile_5  = np.percentile(cumulative_returns, 5,  axis=0)
    percentile_25 = np.percentile(cumulative_returns, 25, axis=0)
    percentile_50 = np.percentile(cumulative_returns, 50, axis=0)
    percentile_75 = np.percentile(cumulative_returns, 75, axis=0)
    percentile_95 = np.percentile(cumulative_returns, 95, axis=0)

    return {
        "final_returns":          cumulative_returns[:, -1],
        "paths":                  cumulative_returns,
        "percentiles": {
            "5":  percentile_5,
            "25": percentile_25,
            "50": percentile_50,
            "75": percentile_75,
            "95": percentile_95,
        },
        "sharpe_ratios":          sharpe_ratios,
        "max_drawdowns":          max_drawdowns,
        "max_drawdown_durations": max_drawdown_durations,
        "total_drawdown_days":    total_drawdown_days,
    }


# ---------------------------------------------------------------------------
# 6.2  Drawdown analysis
# ---------------------------------------------------------------------------

def analyze_drawdowns(
    returns,
    output_dir: str,
    period_length: int,
    test_start_date: str,
    test_end_date: str,
    portfolio_name: str,
    dates=None,
) -> dict:
    """
    Analyse drawdown periods in a cumulative-return series and save visualisation.

    Parameters
    ----------
    returns        : list of *percent-scale* cumulative returns (0.0 = start)
    output_dir     : directory to save the output PNG
    period_length  : length of the test period in trading days (used in chart title)
    test_start_date: ISO date string — start of the test period
    test_end_date  : ISO date string — end of the test period
    portfolio_name : used in the saved file name
    dates          : list of date strings matching returns; synthetic if None

    Returns
    -------
    dict with keys:
        max_drawdown, avg_drawdown, total_drawdown_days, significant_drawdown_days,
        avg_drawdown_length, avg_calendar_days, drawdown_periods, significant_periods,
        max_drawdown_duration, max_calendar_duration, drawdown_durations,
        calendar_durations, drawdown_magnitudes, top_significant_periods
    """
    if dates is None:
        start_date = pd.Timestamp(test_start_date)
        dates = [
            (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(len(returns))
        ]

    if len(dates) != len(returns):
        min_length = min(len(dates), len(returns))
        dates = dates[:min_length]
        returns = returns[:min_length]

    date_objects = [pd.Timestamp(d).date() for d in dates]

    # --- Drawdown period detection -------------------------------------------
    drawdown_periods = []
    drawdowns = []

    running_peak = returns[0]
    current_drawdown_start_idx = None
    in_drawdown = False
    max_drawdown = 0
    max_drawdown_idx = 0

    for i, value in enumerate(returns):
        if value > running_peak:
            running_peak = value

            if in_drawdown:
                min_value = min(returns[current_drawdown_start_idx:i])
                drawdown_depth = (running_peak - min_value) / (1 + running_peak / 100)

                drawdown_periods.append({
                    "start_idx":    current_drawdown_start_idx,
                    "end_idx":      i,
                    "start_date":   dates[current_drawdown_start_idx],
                    "end_date":     dates[i],
                    "duration":     i - current_drawdown_start_idx,
                    "max_drawdown": drawdown_depth,
                    "calendar_days": (
                        date_objects[i] - date_objects[current_drawdown_start_idx]
                    ).days,
                })

                in_drawdown = False
                current_drawdown_start_idx = None

        current_drawdown = (running_peak - value) / (1 + running_peak / 100)
        drawdowns.append(current_drawdown)

        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_idx = i

        if current_drawdown > 0 and not in_drawdown:
            in_drawdown = True
            current_drawdown_start_idx = i

    if in_drawdown:
        min_value = min(returns[current_drawdown_start_idx:])
        drawdown_depth = (running_peak - min_value) / (1 + running_peak / 100)
        drawdown_periods.append({
            "start_idx":    current_drawdown_start_idx,
            "end_idx":      len(returns) - 1,
            "start_date":   dates[current_drawdown_start_idx],
            "end_date":     dates[-1],
            "duration":     len(returns) - current_drawdown_start_idx,
            "max_drawdown": drawdown_depth,
            "calendar_days": (
                date_objects[-1] - date_objects[current_drawdown_start_idx]
            ).days,
        })

    # Recalculate from the drawdowns list for consistency
    if drawdowns:
        max_drawdown = max(drawdowns)
        max_drawdown_idx = drawdowns.index(max_drawdown)

    for i, period in enumerate(drawdown_periods):
        start_idx = period["start_idx"]
        end_idx = period["end_idx"]
        period_drawdowns = drawdowns[start_idx : end_idx + 1]
        if period_drawdowns:
            drawdown_periods[i]["max_drawdown"] = max(period_drawdowns)

    drawdown_periods.sort(key=lambda x: x["max_drawdown"], reverse=True)

    total_days_in_drawdown = sum(p["duration"] for p in drawdown_periods)
    significant_drawdown_days = sum(
        p["calendar_days"] for p in drawdown_periods if p["calendar_days"] > 20
    )

    significant_periods = sorted(
        [p for p in drawdown_periods if p["calendar_days"] > 20],
        key=lambda x: x["max_drawdown"],
        reverse=True,
    )


    avg_drawdown_length = (
        total_days_in_drawdown / len(drawdown_periods) if drawdown_periods else 0
    )
    avg_calendar_days = (
        sum(p["calendar_days"] for p in drawdown_periods) / len(drawdown_periods)
        if drawdown_periods else 0
    )
    non_zero_drawdowns = [d for d in drawdowns if d > 0]
    avg_drawdown_depth = (
        sum(non_zero_drawdowns) / len(non_zero_drawdowns) if non_zero_drawdowns else 0
    )

    # --- Plot ----------------------------------------------------------------
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(range(len(drawdowns)), drawdowns, 0, color="red", alpha=0.3)
    ax1.plot(range(len(drawdowns)), drawdowns, color="red", linewidth=1)
    ax1.set_title(f"Drawdown Over Time - {period_length} days")
    ax1.set_ylabel("Drawdown (%)")
    ax1.set_xlabel("Trading Days")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1f}%"))
    ax1.axhline(y=max_drawdown, color="darkred", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"Maximum Drawdown: {max_drawdown:.2f}%\n"
        f"Average Drawdown: {avg_drawdown_depth:.2f}%\n"
        f"Avg Trading Days: {avg_drawdown_length:.1f}\n"
        f"Avg Calendar Days: {avg_calendar_days:.1f}",
        xy=(max_drawdown_idx, max_drawdown),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.8),
    )

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(returns)), returns, color="blue", linewidth=1.5)
    for period in drawdown_periods:
        ax2.axvspan(
            period["start_idx"], period["end_idx"],
            alpha=0.2, color="red",
            label="_" * period["start_idx"],
        )
    ax2.set_title(
        f"Cumulative Return with Drawdown Periods - {period_length} days"
    )
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_xlabel("Trading Days")
    ax2.grid(True, linestyle="--", alpha=0.7)

    ax3 = fig.add_subplot(gs[1, 0])
    durations = [p["duration"] for p in drawdown_periods]
    calendar_days = [p["calendar_days"] for p in drawdown_periods]
    max_drawdowns_periods = [p["max_drawdown"] for p in drawdown_periods]

    if durations:
        x = np.arange(len(durations))
        width = 0.35
        ax3.bar(x - width / 2, durations, width, color="blue", alpha=0.7, label="Trading Days")
        ax3.bar(x + width / 2, calendar_days, width, color="green", alpha=0.7, label="Calendar Days")
        for i, (_, cal_days, dd) in enumerate(zip(durations, calendar_days, max_drawdowns_periods)):
            ax3.text(i, cal_days + 1, f"{dd:.1f}%", ha="center", va="bottom",
                     color="black", fontsize=8)
        ax3.set_title("Drawdown Duration by Episode")
        ax3.set_ylabel("Duration (Days)")
        ax3.set_xlabel("Drawdown Episode")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "No drawdown periods found", ha="center", va="center", fontsize=12)

    ax4 = fig.add_subplot(gs[1, 1])
    if non_zero_drawdowns:
        sns.histplot(non_zero_drawdowns, bins=20, kde=True, ax=ax4, color="green")
        mean_dd   = np.mean(non_zero_drawdowns)
        median_dd = np.median(non_zero_drawdowns)
        ax4.axvline(mean_dd,     color="red",   linestyle="--", label=f"Mean: {mean_dd:.2f}%")
        ax4.axvline(median_dd,   color="blue",  linestyle="--", label=f"Median: {median_dd:.2f}%")
        ax4.axvline(max_drawdown, color="black", linestyle="-",  label=f"Max: {max_drawdown:.2f}%")
        ax4.set_title("Drawdown Magnitude Distribution")
        ax4.set_xlabel("Drawdown (%)")
        ax4.set_ylabel("Frequency")
        ax4.legend()
        ax4.grid(True, linestyle="--", alpha=0.7)
    else:
        ax4.text(0.5, 0.5, "No non-zero drawdowns found", ha="center", va="center", fontsize=12)

    ax5 = fig.add_subplot(gs[2, 0])
    if durations:
        sns.histplot(durations, bins=min(20, len(durations)), kde=True, ax=ax5,
                     color="blue", alpha=0.4, label="Trading Days")
        sns.histplot(calendar_days, bins=min(20, len(calendar_days)), kde=True, ax=ax5,
                     color="green", alpha=0.4, label="Calendar Days")
        mean_calendar   = np.mean(calendar_days)
        median_calendar = np.median(calendar_days)
        max_calendar    = max(calendar_days)
        ax5.axvline(mean_calendar,   color="darkgreen", linestyle="--",
                    label=f"Mean Calendar: {mean_calendar:.1f} days")
        ax5.axvline(median_calendar, color="green",     linestyle=":",
                    label=f"Median Calendar: {median_calendar:.1f} days")
        ax5.axvline(max_calendar,    color="green",     linestyle="-",
                    label=f"Max Calendar: {max_calendar:.0f} days")
        ax5.set_title("Drawdown Duration Distribution")
        ax5.set_xlabel("Duration (Days)")
        ax5.set_ylabel("Frequency")
        ax5.legend()
        ax5.grid(True, linestyle="--", alpha=0.7)
    else:
        ax5.text(0.5, 0.5, "No drawdown periods found", ha="center", va="center", fontsize=12)

    ax6 = fig.add_subplot(gs[2, 1])
    if durations and max_drawdowns_periods:
        ax6.scatter(max_drawdowns_periods, calendar_days, alpha=0.7, c="green", s=50,
                    label="Calendar Days")
        ax6.scatter(max_drawdowns_periods, durations,     alpha=0.7, c="blue",  s=30,
                    label="Trading Days")
        if len(calendar_days) > 1:
            z = np.polyfit(max_drawdowns_periods, calendar_days, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(max_drawdowns_periods), max(max_drawdowns_periods), 100)
            ax6.plot(x_range, p(x_range), "g--", alpha=0.7)
            with np.errstate(invalid="ignore", divide="ignore"):
                corr = np.corrcoef(max_drawdowns_periods, calendar_days)[0, 1]
            ax6.text(0.05, 0.95, f"Calendar Day Correlation: {corr:.2f}",
                     transform=ax6.transAxes, fontsize=10, verticalalignment="top",
                     bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
        ax6.set_title("Drawdown Magnitude vs Duration")
        ax6.set_xlabel("Maximum Drawdown (%)")
        ax6.set_ylabel("Duration (Days)")
        ax6.grid(True, linestyle="--", alpha=0.7)
        ax6.legend()
    else:
        ax6.text(0.5, 0.5, "No drawdown periods found", ha="center", va="center", fontsize=12)

    plt.suptitle(
        f"Drawdown Analysis ({test_start_date} to {test_end_date})", fontsize=16, y=0.99
    )
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3)
    fig.subplots_adjust(top=0.92)

    save_path = os.path.join(
        output_dir, f"{portfolio_name}_drawdown_analysis_{period_length}d.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "max_drawdown":              max_drawdown,
        "avg_drawdown":              avg_drawdown_depth,
        "total_drawdown_days":       total_days_in_drawdown,
        "significant_drawdown_days": significant_drawdown_days,
        "avg_drawdown_length":       avg_drawdown_length,
        "avg_calendar_days":         avg_calendar_days,
        "drawdown_periods":          len(drawdown_periods),
        "significant_periods":       len(significant_periods),
        "max_drawdown_duration": (
            max(p["duration"] for p in drawdown_periods) if drawdown_periods else 0
        ),
        "max_calendar_duration": (
            max(p["calendar_days"] for p in drawdown_periods) if drawdown_periods else 0
        ),
        "drawdown_durations":   [p["duration"]     for p in drawdown_periods],
        "calendar_durations":   [p["calendar_days"] for p in drawdown_periods],
        "drawdown_magnitudes":  [p["max_drawdown"]  for p in drawdown_periods],
        "top_significant_periods": significant_periods[:5],
    }


# ---------------------------------------------------------------------------
# 6.3  Drawdown distribution plots
# ---------------------------------------------------------------------------

def plot_drawdown_distributions(
    simulation_results: dict,
    actual_max_drawdown: float,
    actual_dd_duration: float,
    period_length: int,
    output_dir: str,
    portfolio_name: str,
) -> dict:
    """
    Plot side-by-side distributions of simulated drawdown magnitude and duration
    against the actual observed values.

    Parameters
    ----------
    simulation_results  : output of run_monte_carlo_simulation()
    actual_max_drawdown : observed max drawdown in percent-scale
    actual_dd_duration  : observed max drawdown duration in calendar days
    period_length       : length of test period in trading days (for chart title)
    output_dir          : directory to save PNG
    portfolio_name      : used in saved file name

    Returns
    -------
    dict with keys:
        dd_mean, dd_median, dd_std, dd_5th, dd_95th, dd_percentile,
        dur_mean, dur_median, dur_std, dur_5th, dur_95th, dur_percentile
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Drawdown magnitude --------------------------------------------------
    max_drawdowns = simulation_results["max_drawdowns"]

    sns.histplot(max_drawdowns, kde=True, bins=30, ax=ax1, color="blue", alpha=0.6)
    ax1.axvline(
        x=actual_max_drawdown, color="r", linestyle="--",
        label=f"Actual: {actual_max_drawdown:.2f}%",
    )

    dd_percentile = stats.percentileofscore(max_drawdowns, actual_max_drawdown)
    ax1.text(
        0.05, 0.95, f"Actual Percentile: {dd_percentile:.1f}%",
        transform=ax1.transAxes, fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    dd_mean   = np.mean(max_drawdowns)
    dd_median = np.median(max_drawdowns)
    dd_std    = np.std(max_drawdowns)
    dd_5th    = np.percentile(max_drawdowns, 5)
    dd_95th   = np.percentile(max_drawdowns, 95)

    stats_text = (
        f"Mean: {dd_mean:.2f}%\n"
        f"Median: {dd_median:.2f}%\n"
        f"Std Dev: {dd_std:.2f}%\n"
        f"5th %ile: {dd_5th:.2f}%\n"
        f"95th %ile: {dd_95th:.2f}%"
    )
    ax1.text(
        0.95, 0.95, stats_text, transform=ax1.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax1.set_title(
        f"Maximum Drawdown Distribution - {period_length} Day Forward Test", fontsize=14
    )
    ax1.set_xlabel("Maximum Drawdown (%)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- Drawdown duration ---------------------------------------------------
    max_dd_durations = simulation_results["max_drawdown_durations"]
    # Scale trading days → estimated calendar days (252 / 365 ≈ 0.69 → inverse ≈ 1.45)
    estimated_calendar_durations = [d * 1.45 for d in max_dd_durations]

    sns.histplot(
        estimated_calendar_durations, kde=True, bins=30, ax=ax2,
        color="green", alpha=0.6, label="Estimated Calendar Days",
    )
    ax2.axvline(
        x=actual_dd_duration, color="r", linestyle="--",
        label=f"Actual: {actual_dd_duration} calendar days",
    )

    duration_percentile = stats.percentileofscore(
        estimated_calendar_durations, actual_dd_duration
    )
    ax2.text(
        0.05, 0.95, f"Actual Percentile: {duration_percentile:.1f}%",
        transform=ax2.transAxes, fontsize=12, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    dur_mean   = np.mean(estimated_calendar_durations)
    dur_median = np.median(estimated_calendar_durations)
    dur_std    = np.std(estimated_calendar_durations)
    dur_5th    = np.percentile(estimated_calendar_durations, 5)
    dur_95th   = np.percentile(estimated_calendar_durations, 95)

    stats_text = (
        f"Mean: {dur_mean:.1f} days\n"
        f"Median: {dur_median:.1f} days\n"
        f"Std Dev: {dur_std:.1f} days\n"
        f"5th %ile: {dur_5th:.1f} days\n"
        f"95th %ile: {dur_95th:.1f} days"
    )
    ax2.text(
        0.95, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
        verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax2.set_title(
        f"Maximum Drawdown Duration Distribution - {period_length} Day Forward Test",
        fontsize=14,
    )
    ax2.set_xlabel("Maximum Drawdown Duration (Calendar Days)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    save_path = os.path.join(
        output_dir, f"{portfolio_name}_drawdown_distributions_{period_length}d.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return {
        "dd_mean":          dd_mean,
        "dd_median":        dd_median,
        "dd_std":           dd_std,
        "dd_5th":           dd_5th,
        "dd_95th":          dd_95th,
        "dd_percentile":    dd_percentile,
        "dur_mean":         dur_mean,
        "dur_median":       dur_median,
        "dur_std":          dur_std,
        "dur_5th":          dur_5th,
        "dur_95th":         dur_95th,
        "dur_percentile":   duration_percentile,
    }


# ---------------------------------------------------------------------------
# 6.4  Walk-forward test
# ---------------------------------------------------------------------------

def run_walk_forward_test(
    dates,
    returns,
    test_period_length: int,
    output_dir: str,
    portfolio_name: str,
) -> dict | None:
    """
    Split historical returns into training and test windows, run MC simulation on
    training data, then compare the MC forecast against the actual test path.

    Parameters
    ----------
    dates             : list of date strings aligned with returns
    returns           : list/array of *percent-scale* daily returns
    test_period_length: number of trading days in the test window
    output_dir        : directory to save PNG outputs
    portfolio_name    : used in saved file names

    Returns
    -------
    dict of results, or None if there is insufficient data.

    Result keys (always present):
        period_length, test_start_date, test_end_date,
        actual_final_return, actual_max_drawdown,
        actual_dd_duration_trading, actual_dd_duration_calendar,
        actual_percentile, median_forecast, forecast_error, percent_error,
        in_90_interval, in_50_interval,
        avg_drawdown, total_drawdown_days, drawdown_periods,
        avg_drawdown_length_trading, avg_drawdown_length_calendar

    Additional keys for test_period_length >= 20:
        actual_annualized_return, actual_sharpe

    Additional keys for test_period_length >= 63:
        dd_mean, dd_median, dd_std, dd_5th, dd_95th, dd_percentile,
        dur_mean, dur_median, dur_std, dur_5th, dur_95th, dur_percentile

    Additional keys for test_period_length >= 252:
        cagr_mean, cagr_median, cagr_std, cagr_5th, cagr_95th,
        actual_cagr, cagr_percentile
    """
    if len(returns) <= test_period_length:
        return None

    train_returns = returns[:-test_period_length]
    test_returns  = returns[-test_period_length:]
    test_dates    = dates[-test_period_length:]

    if len(train_returns) < 30:
        return None

    num_simulations = 10000
    simulation_results = run_monte_carlo_simulation(
        train_returns, num_simulations, test_period_length, annual_periods=252
    )

    # Build actual cumulative return path (percent-scale)
    actual_returns = [0.0]
    cumulative_return = 0.0
    for r in test_returns:
        r_decimal = r / 100.0
        cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
        actual_returns.append(cumulative_return)

    actual_final_return = actual_returns[-1]

    test_start_date = dates[-test_period_length]
    test_end_date   = dates[-1]

    drawdown_stats = analyze_drawdowns(
        actual_returns,
        output_dir,
        test_period_length,
        test_start_date,
        test_end_date,
        portfolio_name,
        dates=[test_dates[0]] + list(test_dates),
    )

    actual_max_drawdown  = drawdown_stats["max_drawdown"]
    actual_max_dd_duration = drawdown_stats["max_drawdown_duration"]

    if test_period_length >= 63:
        dd_distribution_stats = plot_drawdown_distributions(
            simulation_results,
            actual_max_drawdown,
            actual_max_dd_duration,
            test_period_length,
            output_dir,
            portfolio_name,
        )

    if test_period_length >= 20:
        actual_years = test_period_length / 252
        actual_annualized_return = (
            (1 + actual_final_return / 100) ** (1 / actual_years) - 1
        ) * 100
        actual_volatility = np.std(test_returns) * np.sqrt(252)
        actual_sharpe = (
            (actual_annualized_return / actual_volatility)
            if actual_volatility != 0 else 0
        )
    else:
        actual_annualized_return = actual_final_return
        actual_sharpe = 0

    final_returns    = simulation_results["final_returns"]
    actual_percentile = stats.percentileofscore(final_returns, actual_final_return)

    # --- Walk-forward overlay plot -------------------------------------------
    plt.figure(figsize=(12, 8))

    percentiles = simulation_results["percentiles"]
    x = range(len(percentiles["50"]))
    plt.fill_between(
        x, percentiles["5"], percentiles["95"],
        color="lightblue", alpha=0.3, label="5th-95th Percentile",
    )
    plt.fill_between(
        x, percentiles["25"], percentiles["75"],
        color="blue", alpha=0.3, label="25th-75th Percentile",
    )
    plt.plot(x, percentiles["50"], "b-", linewidth=2, label="Median Path")
    plt.plot(
        x, actual_returns, "orange", linewidth=3,
        label=f"Actual ({actual_final_return:.2f}%, {actual_percentile:.1f}%ile)",
    )

    if test_period_length <= 63:
        period_desc = f"{test_period_length} days (~3 months)"
    elif test_period_length <= 126:
        period_desc = f"{test_period_length} days (~6 months)"
    elif test_period_length <= 252:
        period_desc = f"{test_period_length} days (~1 year)"
    elif test_period_length <= 504:
        period_desc = f"{test_period_length} days (~2 years)"
    else:
        period_desc = f"{test_period_length} days"

    plt.title(
        f"Walk-Forward Test: {period_desc} ({test_start_date} to {test_end_date})",
        fontsize=14,
    )
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    save_path = os.path.join(
        output_dir, f"{portfolio_name}_walk_forward_{test_period_length}d.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.close()

    # --- CAGR distribution plot (periods >= 1 year) --------------------------
    if test_period_length >= 252:
        years = test_period_length / 252
        cagr_values = [
            ((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns
        ]
        actual_cagr = ((1 + actual_final_return / 100) ** (1 / years) - 1) * 100

        plt.figure(figsize=(10, 6))
        sns.histplot(cagr_values, kde=True, bins=50)
        plt.axvline(
            x=actual_cagr, color="r", linestyle="--",
            label=f"Actual CAGR: {actual_cagr:.2f}%",
        )
        cagr_percentile = stats.percentileofscore(cagr_values, actual_cagr)
        plt.text(
            0.05, 0.95, f"Actual CAGR Percentile: {cagr_percentile:.1f}%",
            transform=plt.gca().transAxes, fontsize=12, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        plt.title(f"CAGR Distribution - {period_desc} Forward Test", fontsize=14)
        plt.xlabel("CAGR (%)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        cagr_plot_path = os.path.join(
            output_dir, f"{portfolio_name}_cagr_distribution_{test_period_length}d.png"
        )
        plt.savefig(cagr_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        cagr_mean   = np.mean(cagr_values)
        cagr_median = np.median(cagr_values)
        cagr_std    = np.std(cagr_values)
        cagr_5th    = np.percentile(cagr_values, 5)
        cagr_95th   = np.percentile(cagr_values, 95)

    median_return = percentiles["50"][-1]
    error         = actual_final_return - median_return
    percent_error = (error / abs(median_return)) * 100 if abs(median_return) > 0.01 else 0

    in_90_interval = percentiles["5"][-1]  <= actual_final_return <= percentiles["95"][-1]
    in_50_interval = percentiles["25"][-1] <= actual_final_return <= percentiles["75"][-1]

    result = {
        "period_length":              test_period_length,
        "test_start_date":            test_start_date,
        "test_end_date":              test_end_date,
        "actual_final_return":        actual_final_return,
        "actual_annualized_return":   actual_annualized_return if test_period_length >= 20 else None,
        "actual_sharpe":              actual_sharpe if test_period_length >= 20 else None,
        "actual_max_drawdown":        actual_max_drawdown,
        "actual_dd_duration_trading": drawdown_stats["max_drawdown_duration"],
        "actual_dd_duration_calendar": actual_max_dd_duration,
        "actual_percentile":          actual_percentile,
        "median_forecast":            median_return,
        "forecast_error":             error,
        "percent_error":              percent_error,
        "in_90_interval":             in_90_interval,
        "in_50_interval":             in_50_interval,
        "avg_drawdown":               drawdown_stats["avg_drawdown"],
        "total_drawdown_days":        drawdown_stats["total_drawdown_days"],
        "drawdown_periods":           drawdown_stats["drawdown_periods"],
        "avg_drawdown_length_trading":  drawdown_stats["avg_drawdown_length"],
        "avg_drawdown_length_calendar": drawdown_stats["avg_calendar_days"],
    }

    if test_period_length >= 252:
        result.update({
            "cagr_mean":      cagr_mean,
            "cagr_median":    cagr_median,
            "cagr_std":       cagr_std,
            "cagr_5th":       cagr_5th,
            "cagr_95th":      cagr_95th,
            "actual_cagr":    actual_cagr,
            "cagr_percentile": cagr_percentile,
        })

    if test_period_length >= 63:
        result.update({
            "dd_mean":        dd_distribution_stats["dd_mean"],
            "dd_median":      dd_distribution_stats["dd_median"],
            "dd_std":         dd_distribution_stats["dd_std"],
            "dd_5th":         dd_distribution_stats["dd_5th"],
            "dd_95th":        dd_distribution_stats["dd_95th"],
            "dd_percentile":  dd_distribution_stats["dd_percentile"],
            "dur_mean":       dd_distribution_stats["dur_mean"],
            "dur_median":     dd_distribution_stats["dur_median"],
            "dur_std":        dd_distribution_stats["dur_std"],
            "dur_5th":        dd_distribution_stats["dur_5th"],
            "dur_95th":       dd_distribution_stats["dur_95th"],
            "dur_percentile": dd_distribution_stats["dur_percentile"],
        })

    return result


# ---------------------------------------------------------------------------
# 6.5  Pipeline interfaces
# ---------------------------------------------------------------------------

def run_mc_for_signal(
    signal_col: np.ndarray,
    target_returns_moc: np.ndarray,
    bil_returns: np.ndarray,
    dates,
    output_dir: str,
    portfolio_name: str,
    test_period_lengths: list = None,
    num_simulations: int = 10000,
) -> list:
    """
    Pipeline entry point: run walk-forward MC for a single signal's equity curve.

    Computes the signal-masked daily P&L (signal days in target, off days in BIL),
    converts from decimal to percent-scale, then calls run_walk_forward_test() for
    each requested test-period length.

    Parameters
    ----------
    signal_col          : (n_days,) bool — True on active signal days
    target_returns_moc  : (n_days,) float — MOC-shifted target daily returns (decimal)
    bil_returns         : (n_days,) float — BIL daily returns (decimal)
    dates               : list/array of date strings aligned with returns
    output_dir          : directory for PNG outputs
    portfolio_name      : used in saved file names and console output
    test_period_lengths : list of int trading-day windows to test
                          (default: [63, 126, 252])
    num_simulations     : number of MC paths per test window (default 10 000)

    Returns
    -------
    list of dicts (one per test_period_length), each being the result of
    run_walk_forward_test() — or None entries for windows with insufficient data.
    """
    if test_period_lengths is None:
        test_period_lengths = [63, 126, 252]

    # Signal-masked daily P&L (decimal-scale)
    daily_pnl_decimal = np.where(signal_col, target_returns_moc, bil_returns)

    # Convert decimal → percent scale for MC engine
    daily_pnl_pct = (daily_pnl_decimal * 100).tolist()
    dates_list = list(dates)

    results = []
    for period in test_period_lengths:
        result = run_walk_forward_test(
            dates_list, daily_pnl_pct, period, output_dir, portfolio_name
        )
        results.append(result)

    return results


def run_mc_for_portfolio(
    portfolio_returns: np.ndarray,
    dates,
    output_dir: str,
    portfolio_name: str,
    test_period_lengths: list = None,
    num_simulations: int = 10000,
) -> list:
    """
    Pipeline entry point: run walk-forward MC for a pre-computed portfolio return series.

    Parameters
    ----------
    portfolio_returns    : (n_days,) float — daily portfolio returns (decimal-scale)
    dates                : list/array of date strings aligned with portfolio_returns
    output_dir           : directory for PNG outputs
    portfolio_name       : used in saved file names and console output
    test_period_lengths  : list of int trading-day windows to test
                           (default: [63, 126, 252])
    num_simulations      : number of MC paths per test window (default 10 000)

    Returns
    -------
    list of dicts (one per test_period_length), each being the result of
    run_walk_forward_test() — or None entries for windows with insufficient data.
    """
    if test_period_lengths is None:
        test_period_lengths = [63, 126, 252]

    # Convert decimal → percent scale for MC engine
    returns_pct = (np.asarray(portfolio_returns) * 100).tolist()
    dates_list = list(dates)

    results = []
    for period in test_period_lengths:
        result = run_walk_forward_test(
            dates_list, returns_pct, period, output_dir, portfolio_name
        )
        results.append(result)

    return results
