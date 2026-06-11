"""
RSI Search — simple mean-reversion signal sweep entry point.

Sweeps RSI(window) < threshold and RSI(window) > threshold for every combination
of (signal_ticker, rsi_window, threshold, comparator, target_ticker), computes
in-sample backtest metrics, filters by min_trades and min_win_rate, and writes
a CSV to output/rsi_search_{timestamp}.csv.

Does NOT use combos, validation windows, or Monte Carlo.

Usage
-----
    python rsi_search.py

Config
------
    Edit config.py (RSI_SEARCH_CONFIG) before running.
    Requires TIINGO_API_KEYS in signal_pipeline/.env.
"""

import os
import sys
import itertools
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Project root on sys.path so "from src." imports work when run directly
sys.path.insert(0, str(Path(__file__).parent))

from config import RSI_SEARCH_CONFIG, DATA_DIR, OUTPUT_DIR
from src.data.loader import load_api_keys, check_freshness_and_update
from src.data.alignment import load_multi_ticker_aligned
from src.backtest import prepare_moc_returns
from src.indicators import calculate_rsi
from src.metrics import sharpe, total_return, tail_metrics
from src.signals import make_signal_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_comparator(rsi_series: np.ndarray, comparator: str, threshold: float) -> np.ndarray:
    """
    Apply a comparator to produce a boolean signal column.
    NaN RSI values become False.

    Parameters
    ----------
    rsi_series : (n_days,) float — RSI values (NaN for warmup days)
    comparator : "lt", "gt", "lte", or "gte"
    threshold  : float

    Returns
    -------
    (n_days,) bool
    """
    nan_mask = np.isnan(rsi_series)
    if comparator == "lt":
        result = rsi_series < threshold
    elif comparator == "gt":
        result = rsi_series > threshold
    elif comparator == "lte":
        result = rsi_series <= threshold
    elif comparator == "gte":
        result = rsi_series >= threshold
    else:
        raise ValueError(f"Unknown comparator: {comparator!r}")
    return np.where(nan_mask, False, result).astype(bool)


def _compute_signal_metrics(
    signal: np.ndarray,
    target_returns_moc: np.ndarray,
    bil_returns: np.ndarray,
) -> dict | None:
    """
    Compute in-sample metrics for a single signal column against a single target.

    Parameters
    ----------
    signal             : (n_days,) bool
    target_returns_moc : (n_days,) float — MOC-shifted target daily returns (decimal)
    bil_returns        : (n_days,) float — BIL daily returns (decimal)

    Returns
    -------
    dict of metrics, or None if the signal has no active days.
    """
    n_trades = int(signal.sum())
    if n_trades == 0:
        return None

    sig_ret   = target_returns_moc[signal]   # target returns on signal days
    bil_on_sig = bil_returns[signal]          # BIL returns on same days

    win_rt = float((sig_ret > bil_on_sig).mean())
    bench_median = float(np.median(bil_on_sig))
    tot_ret = float(total_return(sig_ret))
    shrp = float(sharpe(sig_ret))
    tail = tail_metrics(sig_ret)
    tail_conc = float(tail["tail_concentration"])

    return {
        "N_Trades":                n_trades,
        "Win_Rate":                win_rt,
        "Benchmark_Median_Return": bench_median,
        "Benchmark_Negative":      bench_median < 0,
        "Total_Return":            tot_ret,
        "Sharpe":                  shrp,
        "Tail_Concentration":      tail_conc,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_rsi_search(config: dict) -> pd.DataFrame:
    """
    Execute the full RSI search sweep and return the results DataFrame.

    Parameters
    ----------
    config : RSI_SEARCH_CONFIG dict

    Returns
    -------
    pd.DataFrame — one row per (signal, target) pair that passes filters
    """
    signal_tickers  = config["signal_tickers"]
    target_tickers  = config["target_tickers"]
    rsi_windows     = config["rsi_windows"]
    rsi_thresholds  = config["rsi_thresholds"]
    comparators     = config["comparators"]
    benchmark_ticker = config["benchmark_ticker"]
    min_trades      = config["min_trades"]
    min_win_rate    = config["min_win_rate"]
    filter_bench_neg = config.get("filter_benchmark_negative", False)

    # --- Download / freshness check -----------------------------------------
    all_tickers = list(dict.fromkeys(signal_tickers + target_tickers + [benchmark_ticker]))
    print(f"\n[rsi_search] Checking data freshness for {all_tickers}...")
    api_keys = load_api_keys()
    check_freshness_and_update(all_tickers, api_keys, DATA_DIR)

    # --- Load aligned price DataFrame ----------------------------------------
    print("[rsi_search] Loading aligned price data...")
    price_df = load_multi_ticker_aligned(all_tickers, DATA_DIR)
    n_days = len(price_df)
    print(f"[rsi_search] {n_days} aligned trading days")

    # --- Pre-compute MOC-shifted returns for each ticker ---------------------
    target_returns_moc = {}
    for ticker in target_tickers:
        raw = price_df[ticker].pct_change().fillna(0.0).to_numpy()
        target_returns_moc[ticker] = prepare_moc_returns(raw)

    bil_raw = price_df[benchmark_ticker].pct_change().fillna(0.0).to_numpy()
    bil_returns = prepare_moc_returns(bil_raw)

    # --- Pre-compute RSI for each (signal_ticker, window) --------------------
    rsi_cache: dict[tuple, np.ndarray] = {}
    for ticker, window in itertools.product(signal_tickers, rsi_windows):
        key = (ticker, window)
        if key not in rsi_cache:
            series = price_df[ticker]
            rsi_cache[key] = calculate_rsi(series, window).to_numpy(dtype=float)

    # --- Sweep ---------------------------------------------------------------
    total_combos = (
        len(signal_tickers) * len(rsi_windows) * len(rsi_thresholds)
        * len(comparators) * len(target_tickers)
    )
    print(f"[rsi_search] Sweeping {total_combos:,} (signal × window × threshold × comparator × target) combinations...")

    rows = []

    for sig_ticker, window, threshold, comparator in itertools.product(
        signal_tickers, rsi_windows, rsi_thresholds, comparators
    ):
        rsi_vals = rsi_cache[(sig_ticker, window)]
        signal = _apply_comparator(rsi_vals, comparator, threshold)

        signal_name = make_signal_name(
            lhs_fn=     "RSI",
            lhs_window= window,
            lhs_ticker= sig_ticker,
            comparator= comparator,
            rhs_type=   "fixed",
            rhs_value=  float(threshold),
        )

        # Compute metrics per target, collect for Best_Target_IS
        per_target: dict[str, dict] = {}
        for target in target_tickers:
            m = _compute_signal_metrics(signal, target_returns_moc[target], bil_returns)
            if m is not None:
                per_target[target] = m

        if not per_target:
            continue

        # Best_Target_IS = target with highest in-sample Sharpe
        best_target = max(per_target, key=lambda t: per_target[t]["Sharpe"])

        for target, metrics in per_target.items():
            # Apply filters
            if metrics["N_Trades"] < min_trades:
                continue
            if metrics["Win_Rate"] < min_win_rate:
                continue
            if filter_bench_neg and metrics["Benchmark_Negative"]:
                continue

            rows.append({
                "Signal":                  signal_name,
                "Target":                  target,
                "Win_Rate":                round(metrics["Win_Rate"], 4),
                "N_Trades":                metrics["N_Trades"],
                "Benchmark_Median_Return": round(metrics["Benchmark_Median_Return"], 6),
                "Benchmark_Negative":      metrics["Benchmark_Negative"],
                "Total_Return":            round(metrics["Total_Return"], 4),
                "Sharpe":                  round(metrics["Sharpe"], 4),
                "Tail_Concentration":      round(metrics["Tail_Concentration"], 4),
                "Best_Target_IS":          best_target,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[rsi_search] Warning: no rows passed the filters.")
        return df

    # Sort by Sharpe descending
    df = df.sort_values("Sharpe", ascending=False).reset_index(drop=True)
    print(f"[rsi_search] {len(df):,} rows passed filters.")
    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results = run_rsi_search(RSI_SEARCH_CONFIG)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"rsi_search_{timestamp}.csv"
    results.to_csv(out_path, index=False)
    print(f"\n[rsi_search] Output written to {out_path}")
