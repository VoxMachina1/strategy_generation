"""
Tests for src/backtest.py and src/metrics.py — Phase 3: Vectorized Backtesting Engine.

Covers SC-1 through SC-4:
  SC-1: batch_backtest returns correct shape (n_signals,) per metric, under 5s on 7500×315
  SC-2: prepare_moc_returns shifts correctly, does not mutate input
  SC-3: run_parallel_backtests matches direct batch_backtest calls (np.allclose, rtol=1e-9)
  SC-4: tail_metrics returns dict with exactly 6 required keys
"""

import time

import numpy as np
import pytest

from src.backtest import batch_backtest, prepare_moc_returns, run_parallel_backtests
from src.metrics import tail_metrics


# ---------------------------------------------------------------------------
# SC-2: prepare_moc_returns
# ---------------------------------------------------------------------------

def test_prepare_moc_returns():
    """Verify roll, zero sentinel on last element, and non-mutation of input."""
    arr = np.array([0.01, 0.02, 0.03, 0.04])
    result = prepare_moc_returns(arr)

    assert result[0] == pytest.approx(0.02)
    assert result[1] == pytest.approx(0.03)
    assert result[2] == pytest.approx(0.04)
    assert result[-1] == pytest.approx(0.0)

    # Original must not be mutated
    assert arr[-1] == pytest.approx(0.04)


# ---------------------------------------------------------------------------
# SC-1: batch_backtest shape, correctness, and performance
# ---------------------------------------------------------------------------

def test_batch_backtest_shape():
    """7500×315 returns correct shape per metric and completes in under 5 seconds."""
    rng = np.random.default_rng(42)
    sm = rng.random((7500, 315)) > 0.5        # (7500, 315) bool
    tr = rng.standard_normal(7500) * 0.01
    bil = np.full(7500, 0.0001)

    start = time.perf_counter()
    result = batch_backtest(sm, tr, bil)
    elapsed = time.perf_counter() - start

    assert isinstance(result, dict)
    expected_keys = {
        "total_return", "cagr", "sharpe", "smart_sharpe", "sortino",
        "max_drawdown", "calmar", "omega", "win_rate", "profit_factor",
        "recovery_factor", "time_in_market", "n_signal_days",
    }
    assert set(result.keys()) == expected_keys

    for key, val in result.items():
        assert val.shape == (315,), f"metric '{key}' shape {val.shape} != (315,)"

    assert elapsed < 5.0, f"batch_backtest took {elapsed:.2f}s on 7500×315, expected < 5s"


def test_batch_backtest_all_on_signal():
    """Signal always True: total daily return equals target returns for all signals."""
    n = 100
    sm = np.ones((n, 3), dtype=bool)
    tr = np.tile([0.01, -0.005, 0.02], n // 3 + 1)[:n]
    bil = np.zeros(n)

    result = batch_backtest(sm, tr, bil)

    expected_tr = float((1 + tr).prod() - 1)
    np.testing.assert_allclose(result["total_return"], expected_tr, rtol=1e-9)
    np.testing.assert_allclose(result["time_in_market"], 1.0, rtol=1e-9)


def test_batch_backtest_all_off_signal():
    """Signal always False: total daily return equals BIL returns for all signals."""
    n = 100
    sm = np.zeros((n, 3), dtype=bool)
    tr = np.random.default_rng(0).standard_normal(n) * 0.01
    bil = np.full(n, 0.0002)

    result = batch_backtest(sm, tr, bil)

    expected_bil_tr = float((1 + bil).prod() - 1)
    np.testing.assert_allclose(result["total_return"], expected_bil_tr, rtol=1e-9)
    np.testing.assert_allclose(result["time_in_market"], 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Metric correctness: sharpe and sortino
# ---------------------------------------------------------------------------

def test_sharpe_known_value():
    """Sharpe of constant series is 0.0 (zero std guard). Non-constant case matches formula."""
    # Constant returns — std == 0, sharpe must be 0.0
    sm_const = np.ones((252, 1), dtype=bool)
    daily_const = np.full(252, 0.001)
    result_const = batch_backtest(sm_const, daily_const, np.zeros(252))
    assert result_const["sharpe"][0] == pytest.approx(0.0)

    # Non-constant case — first half positive, second half negative
    daily2 = np.concatenate([np.full(126, 0.001), np.full(126, -0.001)])
    sm2 = np.ones((252, 1), dtype=bool)
    result2 = batch_backtest(sm2, daily2, np.zeros(252))
    expected = float(daily2.mean() / daily2.std() * np.sqrt(252))
    assert result2["sharpe"][0] == pytest.approx(expected, rel=1e-5)


def test_sortino_known_value():
    """
    Verify the RMS-downside Sortino approximation produces a sensible value.

    With n_pos positive days and n_neg negative days, the RMS-downside formula gives:
        down_std = sqrt(mean(min(r, 0)^2))
                 = sqrt(n_neg / n_total) * std_neg  (for zero-mean negatives)

    Construct a known series and verify the formula against manual calculation.
    """
    n = 200
    # 50 negative days of -0.01, 150 positive days of 0.005
    neg = np.full(50, -0.01)
    pos = np.full(150, 0.005)
    r = np.concatenate([neg, pos])
    sm = np.ones((n, 1), dtype=bool)

    result = batch_backtest(sm, r, np.zeros(n))
    sortino_val = result["sortino"][0]

    # Manual: down_std = sqrt(mean(r_neg_zero_padded^2))
    r_neg_padded = np.where(r < 0, r, 0.0)
    expected_down_std = float(np.sqrt((r_neg_padded ** 2).mean()))
    expected_sortino = float(r.mean() / expected_down_std * np.sqrt(252))
    assert sortino_val == pytest.approx(expected_sortino, rel=1e-5)

    # Also verify sortino > 0 (mean is positive) and finite
    assert np.isfinite(sortino_val)
    assert sortino_val > 0.0


def test_max_drawdown_known_value():
    """Max drawdown on up-then-down series is positive fraction; flat series gives 0."""
    r_decline = np.array([0.1, 0.1, -0.2, -0.1])
    sm = np.ones((4, 1), dtype=bool)
    dd = batch_backtest(sm, r_decline, np.zeros(4))["max_drawdown"][0]
    assert dd > 0.0
    assert dd < 1.0

    r_flat = np.full(10, 0.001)
    sm_flat = np.ones((10, 1), dtype=bool)
    dd_flat = batch_backtest(sm_flat, r_flat, np.zeros(10))["max_drawdown"][0]
    assert dd_flat == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# SC-4: tail_metrics
# ---------------------------------------------------------------------------

def test_tail_metrics_keys():
    """tail_metrics returns dict with exactly 6 required keys; tail_score in [0, 1]."""
    rng = np.random.default_rng(7)
    r = rng.standard_normal(500) * 0.01
    result = tail_metrics(r)

    expected_keys = {
        "tail_score", "tail_concentration", "excess_kurtosis",
        "base_win_rate", "stripped_win_rate", "wr_delta",
    }
    assert set(result.keys()) == expected_keys
    assert 0.0 <= result["tail_score"] <= 1.0
    assert 0.0 <= result["base_win_rate"] <= 1.0
    assert 0.0 <= result["stripped_win_rate"] <= 1.0
    # excess_kurtosis can be any finite float (negative for platykurtic)
    assert np.isfinite(result["excess_kurtosis"])


# ---------------------------------------------------------------------------
# SC-3: run_parallel_backtests matches direct batch_backtest
# ---------------------------------------------------------------------------

def test_run_parallel_backtests_matches_direct():
    """Parallel results are numerically identical to direct batch_backtest on each slice."""
    rng = np.random.default_rng(99)
    n_days, n_sigs = 500, 20
    sm = rng.random((n_days, n_sigs)) > 0.5
    tr = rng.standard_normal(n_days) * 0.01
    bil = np.full(n_days, 0.0001)
    date_index = np.arange(n_days)

    window_specs = [
        {"start_idx": 0,   "end_idx": 125},
        {"start_idx": 125, "end_idx": 250},
        {"start_idx": 250, "end_idx": 375},
        {"start_idx": 375, "end_idx": 500},
    ]

    parallel_results = run_parallel_backtests(
        window_specs, sm, tr, bil, date_index, n_workers=2
    )

    for i, ws in enumerate(window_specs):
        s, e = ws["start_idx"], ws["end_idx"]
        direct = batch_backtest(sm[s:e], tr[s:e], bil[s:e])
        for key in direct:
            np.testing.assert_allclose(
                parallel_results[i][key], direct[key], rtol=1e-9,
                err_msg=f"window {i}, metric '{key}' mismatch",
            )


# ---------------------------------------------------------------------------
# safe_asset_ticker vs benchmark_ticker separation
# ---------------------------------------------------------------------------

def test_inactive_return_uses_safe_asset_not_benchmark():
    """
    When a signal is inactive, the strategy earns safe_asset returns, not
    benchmark returns. Verify batch_backtest uses bil_returns (safe asset)
    for the inactive leg, not some other series.

    Set up a signal that is NEVER active (all False). Total return should
    equal the safe asset's compounded return exactly — not the benchmark's.
    """
    rng = np.random.default_rng(1)
    n = 500
    safe_returns  = rng.normal(0.0001, 0.0002, n)   # low-vol cash-like
    bench_returns = rng.normal(0.0008, 0.015,  n)   # high-vol equity-like

    signal_matrix = np.zeros((n, 1), dtype=bool)    # never fires
    target_returns = rng.normal(0.001, 0.02, n)     # irrelevant — signal never active

    result = batch_backtest(signal_matrix, target_returns, safe_returns)

    expected_total_return = float((1 + safe_returns).prod() - 1)
    np.testing.assert_allclose(
        result["total_return"][0], expected_total_return, rtol=1e-9,
        err_msg="Inactive signal should compound safe_asset returns, not benchmark returns",
    )

    # Confirm benchmark returns would give a different answer
    bench_total = float((1 + bench_returns).prod() - 1)
    assert abs(result["total_return"][0] - bench_total) > 1e-6, (
        "Safe asset and benchmark returns appear identical — test is not meaningful"
    )


def test_stage_compute_returns_uses_safe_asset(tmp_path):
    """
    _stage_compute_returns reads safe_asset_ticker for bil_returns, not
    benchmark_ticker. Verify by giving them different return series and
    checking which one ends up in the output.
    """
    import pandas as pd
    import main as pipeline

    rng = np.random.default_rng(2)
    n = 300
    dates = pd.date_range("2020-01-02", periods=n, freq="B")

    safe_close  = 100 * np.exp(np.cumsum(rng.normal(0.00005, 0.0002, n)))
    bench_close = 100 * np.exp(np.cumsum(rng.normal(0.0004,  0.015,  n)))
    target_close = 100 * np.exp(np.cumsum(rng.normal(0.001,  0.02,   n)))

    price_df = pd.DataFrame(
        {"BIL": safe_close, "SPY": bench_close, "TQQQ": target_close},
        index=dates,
    )

    cfg = {
        "target_tickers":    ["TQQQ"],
        "benchmark_ticker":  "SPY",
        "safe_asset_ticker": "BIL",
    }

    prog = pipeline._Progress(1)
    _, bil_returns = pipeline._stage_compute_returns(cfg, price_df, prog)

    # bil_returns should match BIL pct_change (MOC-shifted), not SPY
    from src.backtest import prepare_moc_returns
    expected_bil = prepare_moc_returns(price_df["BIL"].pct_change().to_numpy())
    expected_spy = prepare_moc_returns(price_df["SPY"].pct_change().to_numpy())

    np.testing.assert_allclose(bil_returns, expected_bil, rtol=1e-9,
        err_msg="bil_returns should come from safe_asset_ticker (BIL), not benchmark_ticker (SPY)")
    assert not np.allclose(bil_returns, expected_spy), (
        "bil_returns matches SPY — safe_asset/benchmark separation is broken"
    )
