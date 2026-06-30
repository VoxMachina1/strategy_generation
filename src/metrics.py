"""
Performance metrics for the Composer Signal Pipeline.

All functions operate on 1D np.ndarray of daily returns.
No quantstats dependency — pure NumPy except tail_metrics (scipy.stats.kurtosis).

Scalar helpers are used in tests and by downstream callers (e.g. tail_metrics).
The vectorized batch computation lives in src/backtest.py:batch_backtest as
inlined NumPy expressions for performance.
"""

import numpy as np
from scipy import stats


def total_return(r: np.ndarray) -> float:
    """Compounded total return over the period."""
    return float((1 + r).prod() - 1)


def cagr(r: np.ndarray, annual: float = 252) -> float:
    """Compound annual growth rate."""
    tr = total_return(r)
    return float((1 + tr) ** (annual / len(r)) - 1)


def sharpe(r: np.ndarray, annual: float = 252) -> float:
    """Annualized Sharpe ratio. Returns 0.0 if std is zero (constant returns)."""
    std = r.std()
    if std == 0:
        return 0.0
    return float(r.mean() / std * np.sqrt(annual))


def smart_sharpe(r: np.ndarray, annual: float = 252) -> float:
    """
    Sharpe corrected for autocorrelation (first 5 lags).
    Formula: sharpe / sqrt(max(1 + 2 * sum(ac_1..ac_5), 1e-9))
    """
    s = sharpe(r, annual)
    sum_ac = 0.0
    for k in range(1, 6):
        ac = np.corrcoef(r[:-k], r[k:])[0, 1]
        if np.isnan(ac):
            ac = 0.0
        sum_ac += ac
    denom = max(abs(1.0 + 2 * sum_ac), 1e-9)
    return float(s / np.sqrt(denom))


def sortino(r: np.ndarray, annual: float = 252) -> float:
    """
    Annualized Sortino ratio using downside std of negative returns.
    Returns 0.0 if there are no negative days or downside std is zero.
    """
    downside = r[r < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(r.mean() / downside.std() * np.sqrt(annual))


def max_drawdown(r: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown from the cumulative log-return curve.
    Returns a positive fraction (e.g. 0.15 for 15% drawdown).
    Returns 0.0 for flat or consistently rising series.
    """
    log_r = np.log1p(r)
    cum = np.cumsum(log_r)
    running_max = np.maximum.accumulate(cum)
    dd = (running_max - cum).max()
    if dd <= 0:
        return 0.0
    return float(np.expm1(dd))


def calmar(r: np.ndarray, annual: float = 252) -> float:
    """CAGR / max drawdown. Returns inf if max drawdown is zero."""
    md = max_drawdown(r)
    if md == 0.0:
        return float("inf")
    return float(cagr(r, annual) / md)


def omega(r: np.ndarray, threshold: float = 0.0) -> float:
    """
    Omega ratio: probability-weighted gains vs losses above threshold.
    Returns inf if there are no losses below threshold.
    """
    gains = float(np.sum(np.maximum(r - threshold, 0.0)))
    losses = float(np.sum(np.maximum(threshold - r, 0.0)))
    if losses == 0:
        return float("inf")
    return gains / losses


def win_rate(r: np.ndarray) -> float:
    """Fraction of days with positive return."""
    return float((r > 0).mean())


def profit_factor(r: np.ndarray) -> float:
    """Gross gains / gross losses. Returns inf if no losing days."""
    pos = r[r > 0].sum()
    neg = r[r < 0].sum()
    if neg == 0:
        return float("inf")
    return float(pos / abs(neg))


def recovery_factor(r: np.ndarray) -> float:
    """Total return / max drawdown. Returns inf if max drawdown is zero."""
    md = max_drawdown(r)
    if md == 0.0:
        return float("inf")
    return float(total_return(r) / md)


def tail_metrics(r: np.ndarray) -> dict:
    """
    Compute tail risk metrics for a single signal's active-day return series.

    Parameters
    ----------
    r : np.ndarray
        Signal-day returns — the non-zero entries from signal_returns[:, j]
        (days the signal was active). The caller is responsible for filtering
        to active days only; this function does not filter internally.

    Returns
    -------
    dict with keys:
        tail_score          : float in [0, 1] — composite tail risk score
        tail_concentration  : float — fraction of total profit from top 5% of signal days
        excess_kurtosis     : float — Fisher kurtosis (can be negative for platykurtic data)
        base_win_rate       : float — win rate over all active days
        stripped_win_rate   : float — win rate after removing top 5% return days
        wr_delta            : float — base_win_rate - stripped_win_rate

    Sub-score normalization uses fixed ranges (not cross-signal):
        tc_score       = clamp(tail_concentration, 0, 1)
        kurtosis_score = clamp(excess_kurtosis / 10, 0, 1)
        wr_delta_score = clamp(wr_delta / 0.5, 0, 1)
        tail_score     = 0.45 * tc_score + 0.30 * kurtosis_score + 0.25 * wr_delta_score
    """
    base_win_rate = float((r > 0).mean())

    p95 = np.percentile(r, 95)
    top_days = r[r > p95]
    pos_days = r[r > 0]
    if pos_days.sum() == 0:
        tail_concentration = 0.0
    else:
        tail_concentration = float(top_days.sum() / pos_days.sum())

    excess_kurtosis = float(stats.kurtosis(r, fisher=True))

    threshold = np.percentile(r, 95)
    r_stripped = r[r <= threshold]
    stripped_win_rate = float((r_stripped > 0).mean()) if len(r_stripped) > 0 else 0.0
    wr_delta = base_win_rate - stripped_win_rate

    tc_score = min(max(tail_concentration, 0.0), 1.0)
    kurtosis_score = min(max(excess_kurtosis / 10.0, 0.0), 1.0)
    wr_delta_score = min(max(wr_delta / 0.5, 0.0), 1.0)
    tail_score = 0.45 * tc_score + 0.30 * kurtosis_score + 0.25 * wr_delta_score

    return {
        "tail_score": tail_score,
        "tail_concentration": tail_concentration,
        "excess_kurtosis": excess_kurtosis,
        "base_win_rate": base_win_rate,
        "stripped_win_rate": stripped_win_rate,
        "wr_delta": wr_delta,
    }
