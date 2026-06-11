"""
Validation framework for the Composer Signal Pipeline.

Implements three evaluation window types and OOS result aggregation.
All window generation is pure index arithmetic — no data access.

Public API
----------
generate_walk_forward_windows()  — non-overlapping sliding test windows
generate_expanding_windows()     — growing train + sliding test windows
generate_rolling_windows()       — fixed-length train + sliding test windows
run_validation()                 — drive parallel backtesting across all windows
aggregate_oos_results()          — reduce per-window results to per-signal stats
compute_regime_stats()           — contiguous "on" block analysis per signal
"""

import numpy as np
import pandas as pd

from src.backtest import batch_backtest, prepare_moc_returns, run_parallel_backtests


# ---------------------------------------------------------------------------
# 5.1 Window slice generators — pure index arithmetic, no data access
# ---------------------------------------------------------------------------

def generate_walk_forward_windows(
    n_days: int,
    train_size: int,
    test_size: int,
) -> list:
    """
    Walk-forward windows: fixed-length train, non-overlapping test slices.

    Train on days [train_start, train_end), test on days [test_start, test_end).
    Each iteration advances both windows by test_size days.

    Parameters
    ----------
    n_days     : total number of days in the dataset
    train_size : number of training days per window
    test_size  : number of test days per window

    Returns
    -------
    list of {"train_start", "train_end", "test_start", "test_end"} dicts (int indices)
    """
    windows = []
    test_start = train_size
    while test_start + test_size <= n_days:
        windows.append({
            "train_start": test_start - train_size,
            "train_end":   test_start,
            "test_start":  test_start,
            "test_end":    test_start + test_size,
        })
        test_start += test_size
    return windows


def generate_expanding_windows(
    n_days: int,
    initial_train: int,
    test_size: int,
) -> list:
    """
    Expanding windows: growing training window, non-overlapping test slices.

    Iteration k trains on days [0, initial_train + k*test_size),
    tests on days [initial_train + k*test_size, initial_train + (k+1)*test_size).

    Parameters
    ----------
    n_days        : total number of days in the dataset
    initial_train : training window size for the first iteration
    test_size     : number of test days per window (constant)

    Returns
    -------
    list of {"train_start", "train_end", "test_start", "test_end"} dicts (int indices)
    """
    windows = []
    k = 0
    while True:
        train_end = initial_train + k * test_size
        test_end = train_end + test_size
        if test_end > n_days:
            break
        windows.append({
            "train_start": 0,
            "train_end":   train_end,
            "test_start":  train_end,
            "test_end":    test_end,
        })
        k += 1
    return windows


def generate_rolling_windows(
    n_days: int,
    train_size: int,
    test_size: int,
    step: int,
) -> list:
    """
    Rolling windows: fixed-length train and test, sliding by step.

    Parameters
    ----------
    n_days     : total number of days in the dataset
    train_size : number of training days per window (fixed)
    test_size  : number of test days per window (fixed)
    step       : number of days to advance per iteration

    Returns
    -------
    list of {"train_start", "train_end", "test_start", "test_end"} dicts (int indices)
    """
    windows = []
    train_start = 0
    while train_start + train_size + test_size <= n_days:
        train_end = train_start + train_size
        windows.append({
            "train_start": train_start,
            "train_end":   train_end,
            "test_start":  train_end,
            "test_end":    train_end + test_size,
        })
        train_start += step
    return windows


# ---------------------------------------------------------------------------
# 5.2 Per-window evaluation
# ---------------------------------------------------------------------------

def run_validation(
    signal_matrix: np.ndarray,
    signal_names: list,
    signal_metadata: list,
    price_df: pd.DataFrame,
    target_tickers: list,
    bil_returns: np.ndarray,
    window_type: str,
    window_config: dict,
    n_workers: int = None,
) -> pd.DataFrame:
    """
    Run backtesting across all validation windows for all target tickers.

    Parameters
    ----------
    signal_matrix   : (n_days, n_signals) bool
    signal_names    : list[str], parallel to signal_matrix columns
    signal_metadata : list[SignalSpec], parallel to signal_matrix columns
    price_df        : pd.DataFrame — close prices; columns include target_tickers;
                      index is a DatetimeIndex aligned with signal_matrix rows
    target_tickers  : list[str] — tickers to compute returns for
    bil_returns     : np.ndarray (n_days,) — BIL daily returns
    window_type     : "walk_forward", "expanding", or "rolling"
    window_config   : dict — keys depend on window_type:
                      walk_forward: {"train_size", "test_size"}
                      expanding:    {"initial_train", "test_size"}
                      rolling:      {"train_size", "test_size", "step"}
    n_workers       : process pool size (defaults to cpu_count - 1)

    Returns
    -------
    pd.DataFrame with columns:
        signal_name, target, window_iteration,
        test_start_date, test_end_date,
        total_return, cagr, sharpe, smart_sharpe, sortino,
        max_drawdown, calmar, omega, win_rate, profit_factor,
        recovery_factor, time_in_market, n_signal_days
    """
    n_days = signal_matrix.shape[0]
    date_index = price_df.index.to_numpy()

    # Generate window specs
    if window_type == "walk_forward":
        windows = generate_walk_forward_windows(
            n_days,
            window_config["train_size"],
            window_config["test_size"],
        )
    elif window_type == "expanding":
        windows = generate_expanding_windows(
            n_days,
            window_config["initial_train"],
            window_config["test_size"],
        )
    elif window_type == "rolling":
        windows = generate_rolling_windows(
            n_days,
            window_config["train_size"],
            window_config["test_size"],
            window_config["step"],
        )
    else:
        raise ValueError(
            f"Unknown window_type: {window_type!r}. "
            "Must be 'walk_forward', 'expanding', or 'rolling'."
        )

    if not windows:
        return pd.DataFrame()

    # Build test-only window specs (backtesting uses test slice)
    test_window_specs = [
        {"start_idx": w["test_start"], "end_idx": w["test_end"]}
        for w in windows
    ]

    rows = []
    n_signals = len(signal_names)

    for ticker in target_tickers:
        # MOC-shift target returns once per ticker
        raw_returns = price_df[ticker].pct_change().fillna(0.0).to_numpy()
        target_returns_moc = prepare_moc_returns(raw_returns)

        # Parallel backtest across all windows
        window_results = run_parallel_backtests(
            test_window_specs,
            signal_matrix,
            target_returns_moc,
            bil_returns,
            date_index,
            n_workers=n_workers,
        )

        metric_keys = [
            "total_return", "cagr", "sharpe", "smart_sharpe", "sortino",
            "max_drawdown", "calmar", "omega", "win_rate", "profit_factor",
            "recovery_factor", "time_in_market", "n_signal_days",
        ]

        for w_idx, (w, result) in enumerate(zip(windows, window_results)):
            test_start_date = date_index[w["test_start"]]
            test_end_date = date_index[min(w["test_end"] - 1, n_days - 1)]

            for sig_idx in range(n_signals):
                row = {
                    "signal_name":      signal_names[sig_idx],
                    "target":           ticker,
                    "window_iteration": w_idx,
                    "test_start_date":  test_start_date,
                    "test_end_date":    test_end_date,
                }
                for key in metric_keys:
                    row[key] = float(result[key][sig_idx])
                rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5.3 OOS aggregation
# ---------------------------------------------------------------------------

def aggregate_oos_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-window OOS results to one row per (signal_name, target).

    Parameters
    ----------
    results_df : output of run_validation() — one row per (signal, target, window)

    Returns
    -------
    pd.DataFrame with one row per (signal_name, target) and columns:
        signal_name, target,
        Sharpe_p10, Sharpe_p50, Sharpe_p90, Sharpe_IQR, Sharpe_Stripped,
        Return_p50, Return_p10, MaxDD_p90,
        Consistency_Score, N_Iterations,
        Sortino_p50, Calmar_p50
    """
    if results_df.empty:
        return pd.DataFrame()

    agg_rows = []

    for (signal_name, target), group in results_df.groupby(["signal_name", "target"]):
        sharpe_vals = group["sharpe"].to_numpy()
        ret_vals = group["total_return"].to_numpy()
        dd_vals = group["max_drawdown"].to_numpy()
        sortino_vals = group["sortino"].to_numpy()
        calmar_vals = group["calmar"].to_numpy()

        n = len(sharpe_vals)

        # Stripped Sharpe: aggregate OOS Sharpe with best window removed
        if n > 1:
            best_idx = np.argmax(sharpe_vals)
            stripped = np.delete(sharpe_vals, best_idx)
            sharpe_stripped = float(np.median(stripped))
        else:
            sharpe_stripped = float(sharpe_vals[0])

        agg_rows.append({
            "signal_name":      signal_name,
            "target":           target,
            "Sharpe_p10":       float(np.percentile(sharpe_vals, 10)),
            "Sharpe_p50":       float(np.percentile(sharpe_vals, 50)),
            "Sharpe_p90":       float(np.percentile(sharpe_vals, 90)),
            "Sharpe_IQR":       float(np.percentile(sharpe_vals, 75) - np.percentile(sharpe_vals, 25)),
            "Sharpe_Stripped":  sharpe_stripped,
            "Return_p50":       float(np.percentile(ret_vals, 50)),
            "Return_p10":       float(np.percentile(ret_vals, 10)),
            "MaxDD_p90":        float(np.percentile(dd_vals, 90)),
            "Sortino_p50":      float(np.percentile(sortino_vals, 50)),
            "Calmar_p50":       float(np.percentile(calmar_vals, 50)),
            "Consistency_Score": float((sharpe_vals > 0).mean()),
            "N_Iterations":     n,
        })

    return pd.DataFrame(agg_rows)


# ---------------------------------------------------------------------------
# 5.6 Regime statistics
# ---------------------------------------------------------------------------

def compute_regime_stats(
    signal_col: np.ndarray,
    target_returns: np.ndarray,
    regime_type_threshold: int = 20,
) -> dict:
    """
    Identify contiguous "on" blocks (regime episodes) in a boolean signal column.

    Parameters
    ----------
    signal_col             : (n_days,) bool — True on active signal days
    target_returns         : (n_days,) float — daily target returns (NOT MOC-shifted;
                             these are used for regime-level P&L, not exact execution sim)
    regime_type_threshold  : median duration threshold for Type1 vs Type2 classification
                             (default 20 trading days)

    Returns
    -------
    dict with keys:
        Regime_Count          : int
        Regime_Duration_Median: float (trading days)
        Regime_Duration_Max   : int (trading days)
        Regime_Hit_Rate       : float — fraction of regimes with positive total return
        Signal_Type           : "Type1" or "Type2"
    """
    if not signal_col.any():
        return {
            "Regime_Count": 0,
            "Regime_Duration_Median": 0.0,
            "Regime_Duration_Max": 0,
            "Regime_Hit_Rate": 0.0,
            "Signal_Type": "Type1",
        }

    # Detect start/end of contiguous True blocks
    padded = np.concatenate([[False], signal_col.astype(bool), [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]   # exclusive end indices

    durations = ends - starts
    hit_returns = [
        float((1 + target_returns[s:e]).prod() - 1)
        for s, e in zip(starts, ends)
    ]

    median_dur = float(np.median(durations))
    signal_type = "Type2" if median_dur >= regime_type_threshold else "Type1"

    return {
        "Regime_Count":           int(len(starts)),
        "Regime_Duration_Median": median_dur,
        "Regime_Duration_Max":    int(durations.max()),
        "Regime_Hit_Rate":        float(sum(r > 0 for r in hit_returns) / len(hit_returns)),
        "Signal_Type":            signal_type,
    }
