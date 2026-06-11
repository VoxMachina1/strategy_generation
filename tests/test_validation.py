"""
Tests for src/validation.py — Phase 5: Validation Framework.

Covers TASKS.md success criteria:
  SC-1: Window generators produce correct non-overlapping, correctly-sized index dicts
  SC-2: run_validation returns DataFrame with correct shape and columns
  SC-3: aggregate_oos_results produces 1 row per (signal, target) with all required columns
  SC-4: compute_regime_stats correctly identifies contiguous on-blocks and classifies type
"""

import numpy as np
import pandas as pd
import pytest

from src.validation import (
    aggregate_oos_results,
    compute_regime_stats,
    generate_expanding_windows,
    generate_rolling_windows,
    generate_walk_forward_windows,
    run_validation,
)


# ---------------------------------------------------------------------------
# SC-1: Window generators
# ---------------------------------------------------------------------------

class TestWalkForwardWindows:
    def test_basic_shape(self):
        """Three non-overlapping test windows of size 50 from 250-day history."""
        windows = generate_walk_forward_windows(n_days=250, train_size=100, test_size=50)
        assert len(windows) == 3
        for w in windows:
            assert w["test_end"] - w["test_start"] == 50
            assert w["train_end"] - w["train_start"] == 100
            assert w["train_end"] == w["test_start"]

    def test_non_overlapping_test_windows(self):
        """Test windows must not overlap."""
        windows = generate_walk_forward_windows(n_days=500, train_size=100, test_size=100)
        for i in range(len(windows) - 1):
            assert windows[i]["test_end"] == windows[i + 1]["test_start"]

    def test_no_windows_when_too_short(self):
        """Returns empty list when dataset is shorter than train + test."""
        assert generate_walk_forward_windows(50, 100, 50) == []

    def test_required_keys(self):
        windows = generate_walk_forward_windows(200, 100, 50)
        for w in windows:
            assert set(w.keys()) == {"train_start", "train_end", "test_start", "test_end"}


class TestExpandingWindows:
    def test_train_grows(self):
        """Each window's train_end is larger than the previous."""
        windows = generate_expanding_windows(n_days=400, initial_train=100, test_size=50)
        assert len(windows) >= 2
        for i in range(len(windows) - 1):
            assert windows[i + 1]["train_end"] > windows[i]["train_end"]

    def test_train_always_starts_at_zero(self):
        windows = generate_expanding_windows(n_days=300, initial_train=100, test_size=50)
        for w in windows:
            assert w["train_start"] == 0

    def test_non_overlapping_test_windows(self):
        windows = generate_expanding_windows(n_days=400, initial_train=100, test_size=50)
        for i in range(len(windows) - 1):
            assert windows[i]["test_end"] == windows[i + 1]["test_start"]


class TestRollingWindows:
    def test_fixed_train_size(self):
        windows = generate_rolling_windows(n_days=300, train_size=100, test_size=50, step=50)
        for w in windows:
            assert w["train_end"] - w["train_start"] == 100
            assert w["test_end"] - w["test_start"] == 50

    def test_step_advances_correctly(self):
        windows = generate_rolling_windows(n_days=400, train_size=100, test_size=50, step=25)
        for i in range(len(windows) - 1):
            assert windows[i + 1]["train_start"] - windows[i]["train_start"] == 25

    def test_no_windows_beyond_dataset(self):
        windows = generate_rolling_windows(n_days=140, train_size=100, test_size=50, step=10)
        for w in windows:
            assert w["test_end"] <= 140


# ---------------------------------------------------------------------------
# SC-2: run_validation DataFrame shape and columns
# ---------------------------------------------------------------------------

def _make_price_df(n_days, tickers, seed=0):
    """Synthetic price DataFrame for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    data = {}
    for t in tickers:
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
        data[t] = prices
    return pd.DataFrame(data, index=dates)


def test_run_validation_shape():
    """
    3 signals × 2 targets × 5 walk-forward windows = 30 rows.
    All required columns present.
    """
    rng = np.random.default_rng(42)
    n_days = 350
    n_sigs = 3
    tickers = ["TQQQ", "BIL"]

    sm = rng.random((n_days, n_sigs)) > 0.5
    signal_names = [f"SIG_{i}" for i in range(n_sigs)]
    price_df = _make_price_df(n_days, tickers)
    bil = np.full(n_days, 0.0001)

    window_config = {"train_size": 100, "test_size": 50}
    # n_days=350, train=100, test=50 → 5 windows (100..150, 150..200, 200..250, 250..300, 300..350)
    result = run_validation(
        sm, signal_names, [None] * n_sigs,
        price_df, tickers, bil,
        window_type="walk_forward",
        window_config=window_config,
        n_workers=1,
    )

    n_windows = len(generate_walk_forward_windows(n_days, 100, 50))
    expected_rows = n_sigs * len(tickers) * n_windows
    assert len(result) == expected_rows, f"Expected {expected_rows} rows, got {len(result)}"

    required_cols = {
        "signal_name", "target", "window_iteration",
        "test_start_date", "test_end_date",
        "sharpe", "sortino", "total_return", "max_drawdown",
    }
    assert required_cols.issubset(set(result.columns))


def test_run_validation_empty_when_no_windows():
    """Returns empty DataFrame when dataset is too short to form any windows."""
    rng = np.random.default_rng(0)
    sm = rng.random((50, 2)) > 0.5
    price_df = _make_price_df(50, ["TQQQ"])
    bil = np.zeros(50)

    result = run_validation(
        sm, ["A", "B"], [None, None],
        price_df, ["TQQQ"], bil,
        window_type="walk_forward",
        window_config={"train_size": 100, "test_size": 50},
        n_workers=1,
    )
    assert result.empty


# ---------------------------------------------------------------------------
# SC-3: aggregate_oos_results
# ---------------------------------------------------------------------------

def test_aggregate_oos_results_shape():
    """3 signals × 2 targets × 5 windows → 6 aggregated rows."""
    rng = np.random.default_rng(7)
    n_signals, n_targets, n_windows = 3, 2, 5
    rows = []
    for sig in [f"SIG_{i}" for i in range(n_signals)]:
        for tgt in ["TQQQ", "BIL"]:
            for w in range(n_windows):
                rows.append({
                    "signal_name":      sig,
                    "target":           tgt,
                    "window_iteration": w,
                    "test_start_date":  None,
                    "test_end_date":    None,
                    "sharpe":           float(rng.normal(0.5, 1.0)),
                    "sortino":          float(rng.normal(0.6, 1.0)),
                    "calmar":           float(rng.normal(0.3, 0.5)),
                    "total_return":     float(rng.normal(0.02, 0.05)),
                    "max_drawdown":     float(abs(rng.normal(0.1, 0.05))),
                    "cagr":             0.05,
                    "omega":            1.2,
                    "win_rate":         0.55,
                    "profit_factor":    1.3,
                    "recovery_factor":  1.1,
                    "time_in_market":   0.5,
                    "n_signal_days":    60.0,
                    "smart_sharpe":     0.4,
                })
    df = pd.DataFrame(rows)
    agg = aggregate_oos_results(df)

    assert len(agg) == n_signals * n_targets  # 6 rows
    required_cols = {
        "signal_name", "target",
        "Sharpe_p10", "Sharpe_p50", "Sharpe_p90", "Sharpe_IQR", "Sharpe_Stripped",
        "Return_p50", "Return_p10", "MaxDD_p90",
        "Consistency_Score", "N_Iterations",
    }
    assert required_cols.issubset(set(agg.columns))


def test_aggregate_oos_results_consistency_score():
    """Consistency_Score = fraction of windows with positive Sharpe."""
    rows = [
        {"signal_name": "SIG", "target": "T", "window_iteration": i,
         "test_start_date": None, "test_end_date": None,
         "sharpe": 1.0 if i < 3 else -0.5,
         "sortino": 0.5, "calmar": 0.3,
         "total_return": 0.01, "max_drawdown": 0.05,
         "cagr": 0.05, "omega": 1.1, "win_rate": 0.5,
         "profit_factor": 1.2, "recovery_factor": 1.0,
         "time_in_market": 0.5, "n_signal_days": 50.0, "smart_sharpe": 0.4}
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    agg = aggregate_oos_results(df)
    # 3 of 5 windows have positive Sharpe
    assert agg.iloc[0]["Consistency_Score"] == pytest.approx(3 / 5)
    assert agg.iloc[0]["N_Iterations"] == 5


def test_aggregate_oos_results_empty():
    assert aggregate_oos_results(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# SC-4: compute_regime_stats
# ---------------------------------------------------------------------------

def test_regime_stats_basic():
    """Two contiguous on-blocks are correctly identified."""
    # Block 1: days 2-4 (3 days), Block 2: days 7-9 (3 days)
    sig = np.array([False, False, True, True, True, False, False, True, True, True])
    tr = np.zeros(10)
    result = compute_regime_stats(sig, tr)
    assert result["Regime_Count"] == 2
    assert result["Regime_Duration_Median"] == pytest.approx(3.0)
    assert result["Regime_Duration_Max"] == 3


def test_regime_stats_all_off():
    """All-off signal returns zero counts."""
    sig = np.zeros(50, dtype=bool)
    tr = np.zeros(50)
    result = compute_regime_stats(sig, tr)
    assert result["Regime_Count"] == 0
    assert result["Signal_Type"] == "Type1"


def test_regime_stats_type_classification():
    """Median duration >= 20 → Type2; < 20 → Type1."""
    # Long regime: 25 consecutive days on
    sig_long = np.zeros(100, dtype=bool)
    sig_long[10:35] = True   # 25-day regime → Type2
    result_long = compute_regime_stats(sig_long, np.zeros(100), regime_type_threshold=20)
    assert result_long["Signal_Type"] == "Type2"

    # Short regimes: two 5-day blocks → median 5 → Type1
    sig_short = np.zeros(50, dtype=bool)
    sig_short[5:10] = True
    sig_short[20:25] = True
    result_short = compute_regime_stats(sig_short, np.zeros(50), regime_type_threshold=20)
    assert result_short["Signal_Type"] == "Type1"


def test_regime_stats_hit_rate():
    """Regime hit rate = fraction of regimes with positive total return."""
    sig = np.zeros(20, dtype=bool)
    sig[0:5] = True   # regime 1: returns sum > 0
    sig[10:15] = True  # regime 2: returns sum < 0

    tr = np.zeros(20)
    tr[0:5] = 0.01    # regime 1 wins
    tr[10:15] = -0.01  # regime 2 loses

    result = compute_regime_stats(sig, tr)
    assert result["Regime_Hit_Rate"] == pytest.approx(0.5)
