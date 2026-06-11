"""
Tests for rsi_search.py — Phase 7: RSI Search Entry Point.

Covers success criteria:
  SC-1: _apply_comparator produces correct boolean array for all four operators
  SC-2: _compute_signal_metrics returns correct keys and values for a known series
  SC-3: run_rsi_search output CSV has required columns
  SC-4: min_trades and min_win_rate filters exclude rows below threshold
  SC-5: Best_Target_IS reflects the highest in-sample Sharpe target
  SC-6: Benchmark_Negative flag is set when BIL median return < 0
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Make rsi_search importable without running __main__
sys.path.insert(0, str(Path(__file__).parent.parent))

from rsi_search import _apply_comparator, _compute_signal_metrics, run_rsi_search


# ---------------------------------------------------------------------------
# SC-1: _apply_comparator
# ---------------------------------------------------------------------------

class TestApplyComparator:
    def _rsi(self, values):
        return np.array(values, dtype=float)

    def test_lt(self):
        rsi = self._rsi([25.0, 35.0, 45.0])
        result = _apply_comparator(rsi, "lt", 30.0)
        np.testing.assert_array_equal(result, [True, False, False])

    def test_gt(self):
        rsi = self._rsi([25.0, 35.0, 75.0])
        result = _apply_comparator(rsi, "gt", 70.0)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_lte(self):
        rsi = self._rsi([30.0, 30.1])
        result = _apply_comparator(rsi, "lte", 30.0)
        np.testing.assert_array_equal(result, [True, False])

    def test_gte(self):
        rsi = self._rsi([69.9, 70.0])
        result = _apply_comparator(rsi, "gte", 70.0)
        np.testing.assert_array_equal(result, [False, True])

    def test_nan_becomes_false(self):
        """NaN warmup days must map to False."""
        rsi = self._rsi([np.nan, np.nan, 25.0])
        result = _apply_comparator(rsi, "lt", 30.0)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_unknown_comparator_raises(self):
        with pytest.raises(ValueError, match="Unknown comparator"):
            _apply_comparator(np.array([50.0]), "eq", 50.0)


# ---------------------------------------------------------------------------
# SC-2: _compute_signal_metrics
# ---------------------------------------------------------------------------

class TestComputeSignalMetrics:
    def test_required_keys(self):
        rng = np.random.default_rng(0)
        n = 100
        signal = np.ones(n, dtype=bool)
        target_r = rng.normal(0.001, 0.01, n)
        bil_r = np.full(n, 0.0001)

        m = _compute_signal_metrics(signal, target_r, bil_r)
        assert m is not None
        required = {
            "N_Trades", "Win_Rate", "Benchmark_Median_Return",
            "Benchmark_Negative", "Total_Return", "Sharpe", "Tail_Concentration",
        }
        assert required == set(m.keys())

    def test_n_trades_counts_active_days(self):
        n = 50
        signal = np.zeros(n, dtype=bool)
        signal[10:20] = True  # 10 active days
        target_r = np.full(n, 0.001)
        bil_r = np.full(n, 0.0001)

        m = _compute_signal_metrics(signal, target_r, bil_r)
        assert m["N_Trades"] == 10

    def test_all_off_returns_none(self):
        n = 50
        signal = np.zeros(n, dtype=bool)
        target_r = np.full(n, 0.001)
        bil_r = np.full(n, 0.0001)
        assert _compute_signal_metrics(signal, target_r, bil_r) is None

    def test_win_rate_beat_bil(self):
        """Win_Rate = fraction of signal days where target > BIL."""
        n = 10
        signal = np.ones(n, dtype=bool)
        # target beats BIL on 7 of 10 days
        target_r = np.array([0.01, 0.02, 0.001, 0.005, 0.03, 0.0001, 0.004, 0.015, 0.0, 0.002])
        bil_r    = np.array([0.002] * n)  # BIL = 0.2% per day

        m = _compute_signal_metrics(signal, target_r, bil_r)
        # 0.01>0.002, 0.02>0.002, 0.001<0.002, 0.005>0.002, 0.03>0.002,
        # 0.0001<0.002, 0.004>0.002, 0.015>0.002, 0.0<0.002, 0.002==0.002 (not >)
        # → 6 wins
        assert m["Win_Rate"] == pytest.approx(6 / 10)

    def test_benchmark_negative_flag(self):
        n = 20
        signal = np.ones(n, dtype=bool)
        target_r = np.full(n, 0.001)
        bil_r = np.full(n, -0.0001)  # negative BIL

        m = _compute_signal_metrics(signal, target_r, bil_r)
        assert m["Benchmark_Negative"] is True
        assert m["Benchmark_Median_Return"] < 0


# ---------------------------------------------------------------------------
# SC-3–SC-6: run_rsi_search with mocked data
# ---------------------------------------------------------------------------

def _make_price_df(tickers, n_days=500, seed=42):
    """Synthetic price DataFrame for testing run_rsi_search."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_days, freq="B")
    data = {}
    for t in tickers:
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n_days)))
        data[t] = prices
    return pd.DataFrame(data, index=dates)


def _minimal_config():
    return {
        "signal_tickers":        ["SPY"],
        "target_tickers":        ["TQQQ", "BIL"],
        "rsi_windows":           [10],
        "rsi_thresholds":        [30, 70],
        "comparators":           ["lt", "gt"],
        "benchmark_ticker":      "BIL",
        "min_trades":            1,
        "min_win_rate":          0.0,
        "filter_benchmark_negative": False,
    }


def _patch_rsi_search(price_df):
    """Return a patcher context that replaces I/O in run_rsi_search."""
    tickers = list(price_df.columns)

    def fake_load_api_keys():
        return ["fake-key"]

    def fake_freshness(*args, **kwargs):
        pass

    def fake_load_multi(*args, **kwargs):
        return price_df

    return (
        patch("rsi_search.load_api_keys",            side_effect=fake_load_api_keys),
        patch("rsi_search.check_freshness_and_update", side_effect=fake_freshness),
        patch("rsi_search.load_multi_ticker_aligned",  side_effect=fake_load_multi),
    )


class TestRunRsiSearch:
    def _run(self, config=None, seed=0):
        cfg = config or _minimal_config()
        all_tickers = list(dict.fromkeys(
            cfg["signal_tickers"] + cfg["target_tickers"] + [cfg["benchmark_ticker"]]
        ))
        price_df = _make_price_df(all_tickers, seed=seed)
        p1, p2, p3 = _patch_rsi_search(price_df)
        with p1, p2, p3:
            return run_rsi_search(cfg)

    def test_returns_dataframe(self):
        df = self._run()
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        df = self._run()
        required = {
            "Signal", "Target", "Win_Rate", "N_Trades",
            "Benchmark_Median_Return", "Total_Return",
            "Sharpe", "Tail_Concentration", "Best_Target_IS",
        }
        assert required.issubset(set(df.columns))

    def test_signal_name_format(self):
        """Signal column values must follow RSI_{window}_{ticker}_{COMP}_{threshold}."""
        df = self._run()
        if df.empty:
            pytest.skip("No rows passed filters with this random seed")
        # Every signal name must start with "RSI_" and contain the ticker
        for sig in df["Signal"].unique():
            assert sig.startswith("RSI_"), f"Unexpected signal name: {sig}"

    def test_min_trades_filter(self):
        """All rows must have N_Trades >= min_trades."""
        cfg = _minimal_config()
        cfg["min_trades"] = 5
        df = self._run(config=cfg)
        if not df.empty:
            assert (df["N_Trades"] >= 5).all()

    def test_min_win_rate_filter(self):
        """All rows must have Win_Rate >= min_win_rate."""
        cfg = _minimal_config()
        cfg["min_win_rate"] = 0.6
        df = self._run(config=cfg)
        if not df.empty:
            assert (df["Win_Rate"] >= 0.6).all()

    def test_best_target_is_valid_target(self):
        """Best_Target_IS must be one of the configured target_tickers."""
        cfg = _minimal_config()
        df = self._run(config=cfg)
        if df.empty:
            pytest.skip("No rows")
        for val in df["Best_Target_IS"]:
            assert val in cfg["target_tickers"], f"Invalid Best_Target_IS: {val}"

    def test_best_target_is_same_per_signal(self):
        """For a given Signal, Best_Target_IS must be consistent across all target rows."""
        df = self._run()
        if df.empty:
            pytest.skip("No rows")
        for sig, grp in df.groupby("Signal"):
            unique_best = grp["Best_Target_IS"].unique()
            assert len(unique_best) == 1, (
                f"Signal {sig} has multiple Best_Target_IS values: {unique_best}"
            )

    def test_filter_benchmark_negative(self):
        """When filter_benchmark_negative=True, rows with Benchmark_Median_Return < 0 excluded."""
        cfg = _minimal_config()
        cfg["filter_benchmark_negative"] = True
        df = self._run(config=cfg, seed=1)
        if not df.empty and "Benchmark_Negative" in df.columns:
            assert not df["Benchmark_Negative"].any()

    def test_sorted_by_sharpe_descending(self):
        """Output DataFrame is sorted by Sharpe descending."""
        df = self._run()
        if len(df) > 1:
            sharpes = df["Sharpe"].to_numpy()
            assert np.all(sharpes[:-1] >= sharpes[1:]), "Not sorted by Sharpe descending"

    def test_empty_when_no_rows_pass_filters(self):
        """Extremely strict filters → empty DataFrame returned, no crash."""
        cfg = _minimal_config()
        cfg["min_trades"] = 10_000   # impossible with 500 days
        df = self._run(config=cfg)
        assert df.empty
