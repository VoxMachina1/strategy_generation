"""
Vectorized backtesting engine for the Composer Signal Pipeline.

All metric computation in batch_backtest uses NumPy broadcasting across the full
(n_days, n_signals) matrix — no Python loops over signals, except the explicitly
permitted smart_sharpe autocorrelation loop (5 lags per signal, O(n_signals) outer loop).

MOC execution model (verified correct for Composer):
    Signal at close t → trade executes close t to close t+1.
    Implemented via np.roll(-1) in prepare_moc_returns(), applied exactly once before
    any backtest call. Do NOT add a NEXT_BAR mode.
"""

import os

import numpy as np
from concurrent.futures import ProcessPoolExecutor


# ---------------------------------------------------------------------------
# Worker globals for process pool (set once by _init_worker, read-only after)
# ---------------------------------------------------------------------------

_SIGNAL_MATRIX = None
_TARGET_RETURNS = None
_BIL_RETURNS = None
_DATE_INDEX = None


# ---------------------------------------------------------------------------
# MOC shift
# ---------------------------------------------------------------------------

def prepare_moc_returns(raw_returns: np.ndarray) -> np.ndarray:
    """
    Shift returns by -1 to implement MOC execution.

    Signal at close t → return from close t to close t+1.
    Composer's verified execution model: condition evaluated at 3:50PM,
    trade executes at 4PM close. Apply exactly once before any backtest call.
    Do NOT add a NEXT_BAR mode.

    Returns a new array; raw_returns is not mutated.
    """
    moc = np.roll(raw_returns, -1)
    moc[-1] = 0.0
    return moc


# ---------------------------------------------------------------------------
# Vectorized batch backtest
# ---------------------------------------------------------------------------

def batch_backtest(
    signal_matrix: np.ndarray,
    target_returns_moc: np.ndarray,
    bil_returns: np.ndarray,
) -> dict:
    """
    Vectorized backtest of all signals against one target ticker.

    Parameters
    ----------
    signal_matrix       : np.ndarray, shape (n_days, n_signals), dtype bool
    target_returns_moc  : np.ndarray, shape (n_days,), float — MOC-shifted via
                          prepare_moc_returns(); do not pass raw_returns here
    bil_returns         : np.ndarray, shape (n_days,), float — BIL daily returns
                          held when signal is off

    Returns
    -------
    dict[str, np.ndarray] — each value has shape (n_signals,):
        total_return, cagr, sharpe, smart_sharpe, sortino, max_drawdown,
        calmar, omega, win_rate, profit_factor, recovery_factor,
        time_in_market, n_signal_days
    """
    n_days, n_signals = signal_matrix.shape

    # Core daily P&L construction — no Python loops
    sr = signal_matrix * target_returns_moc[:, np.newaxis]   # (n_days, n_signals)
    bil = (~signal_matrix) * bil_returns[:, np.newaxis]
    td = sr + bil                                             # total_daily

    # --- Total return and CAGR ---
    total_ret = (1 + td).prod(axis=0) - 1                    # (n_signals,)
    cagr_arr = (1 + total_ret) ** (252.0 / n_days) - 1

    # --- Sharpe ---
    td_mean = td.mean(axis=0)
    td_std = td.std(axis=0)
    _std_zero = td_std < 1e-10
    sharpe_arr = np.where(
        _std_zero,
        0.0,
        td_mean / np.where(_std_zero, 1.0, td_std) * np.sqrt(252),
    )

    # --- Smart Sharpe (per-column autocorrelation loop — explicitly permitted) ---
    smart_sharpe_arr = np.empty(n_signals)
    for j in range(n_signals):
        col = td[:, j]
        s = sharpe_arr[j]
        sum_ac = 0.0
        for k in range(1, 6):
            ac = np.corrcoef(col[:-k], col[k:])[0, 1]
            if np.isnan(ac):
                ac = 0.0
            sum_ac += ac
        denom = max(1.0 + 2 * sum_ac, 1e-9)
        smart_sharpe_arr[j] = s / np.sqrt(denom)

    # --- Sortino (RMS of zero-padded negatives — vectorizable approximation) ---
    # downside std = sqrt(mean(min(r, 0)^2)) across all days (including zeros for off-days).
    # This is smaller than std(r[r<0]) by factor sqrt(n_neg/n_days) — a known approx
    # that avoids a per-column loop. Produces higher Sortino values than the classic formula.
    downside = np.where(td < 0, td, 0.0)
    down_std = np.sqrt((downside ** 2).mean(axis=0))
    sortino_arr = np.where(
        down_std == 0,
        0.0,
        td_mean / np.where(down_std == 0, 1.0, down_std) * np.sqrt(252),
    )

    # --- Max drawdown (log-return cumsum approach) ---
    log_td = np.log1p(td)
    cum = np.cumsum(log_td, axis=0)
    running_max = np.maximum.accumulate(cum, axis=0)
    dd_log = (running_max - cum).max(axis=0)
    max_dd = np.expm1(np.maximum(dd_log, 0.0))               # positive fraction

    # --- Calmar ---
    calmar_arr = np.where(max_dd == 0, np.inf, cagr_arr / np.where(max_dd == 0, 1.0, max_dd))

    # --- Omega ---
    gains = np.sum(np.maximum(td, 0.0), axis=0)
    losses = np.sum(np.maximum(-td, 0.0), axis=0)
    omega_arr = np.where(losses == 0, np.inf, gains / np.where(losses == 0, 1.0, losses))

    # --- Win rate (active days only) ---
    n_active = signal_matrix.sum(axis=0).astype(float)
    n_wins = (sr > 0).sum(axis=0).astype(float)
    win_rate_arr = np.where(
        n_active == 0,
        0.0,
        n_wins / np.where(n_active == 0, 1.0, n_active),
    )

    # --- Profit factor ---
    pos_sum = np.sum(np.maximum(sr, 0.0), axis=0)
    neg_sum = np.abs(np.sum(np.minimum(sr, 0.0), axis=0))
    pf_arr = np.where(neg_sum == 0, np.inf, pos_sum / np.where(neg_sum == 0, 1.0, neg_sum))

    # --- Recovery factor ---
    rf_arr = np.where(
        max_dd == 0,
        np.inf,
        total_ret / np.where(max_dd == 0, 1.0, max_dd),
    )

    # --- Time in market and signal day count ---
    time_in_market = signal_matrix.mean(axis=0)
    n_signal_days = n_active  # already computed above

    return {
        "total_return": total_ret,
        "cagr": cagr_arr,
        "sharpe": sharpe_arr,
        "smart_sharpe": smart_sharpe_arr,
        "sortino": sortino_arr,
        "max_drawdown": max_dd,
        "calmar": calmar_arr,
        "omega": omega_arr,
        "win_rate": win_rate_arr,
        "profit_factor": pf_arr,
        "recovery_factor": rf_arr,
        "time_in_market": time_in_market,
        "n_signal_days": n_signal_days,
    }


# ---------------------------------------------------------------------------
# Process pool infrastructure (initializer pattern)
# ---------------------------------------------------------------------------

def _init_worker(signal_matrix, target_returns, bil_returns, date_index):
    """
    Worker initializer: load shared data into module-level globals once at startup.
    Large arrays are transferred to each worker process exactly once, not per task.
    """
    global _SIGNAL_MATRIX, _TARGET_RETURNS, _BIL_RETURNS, _DATE_INDEX
    _SIGNAL_MATRIX = signal_matrix
    _TARGET_RETURNS = target_returns
    _BIL_RETURNS = bil_returns
    _DATE_INDEX = date_index


def _backtest_window(window_spec: dict) -> dict:
    """
    Worker function: slice globals by window bounds and run batch_backtest.
    Receives only the window index bounds — not the data — keeping task serialization tiny.
    """
    start_idx = window_spec["start_idx"]
    end_idx = window_spec["end_idx"]
    sm_slice = _SIGNAL_MATRIX[start_idx:end_idx]
    tr_slice = _TARGET_RETURNS[start_idx:end_idx]
    bil_slice = _BIL_RETURNS[start_idx:end_idx]
    result = batch_backtest(sm_slice, tr_slice, bil_slice)
    result["window_spec"] = window_spec
    return result


def run_parallel_backtests(
    window_specs: list,
    signal_matrix: np.ndarray,
    target_returns: np.ndarray,
    bil_returns: np.ndarray,
    date_index: np.ndarray,
    n_workers: int = None,
) -> list:
    """
    Run batch_backtest on multiple time windows in parallel using a process pool.

    Each window is processed by a separate worker. Large shared data (signal_matrix,
    target_returns, bil_returns, date_index) is transferred to each worker once via
    the initializer pattern — not serialized with every task.

    Parameters
    ----------
    window_specs   : list of {"start_idx": int, "end_idx": int} dicts
    signal_matrix  : np.ndarray, shape (n_days, n_signals), dtype bool
    target_returns : np.ndarray, shape (n_days,), float — MOC-shifted
    bil_returns    : np.ndarray, shape (n_days,), float
    date_index     : np.ndarray, shape (n_days,)
    n_workers      : int, optional — defaults to max(1, cpu_count - 1)

    Returns
    -------
    list[dict] — one result dict per window_spec, in input order
    """
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 2) - 1)

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=_init_worker,
        initargs=(signal_matrix, target_returns, bil_returns, date_index),
    ) as pool:
        return list(pool.map(_backtest_window, window_specs))
