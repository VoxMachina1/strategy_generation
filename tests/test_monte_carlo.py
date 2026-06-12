"""
Tests for src/monte_carlo.py — Phase 6: Monte Carlo Integration.

Covers success criteria:
  SC-1: run_monte_carlo_simulation returns correct keys and shapes
  SC-2: analyze_drawdowns returns correct stats for a known series
  SC-3: plot_drawdown_distributions returns correct keys (matplotlib mocked)
  SC-4: run_walk_forward_test returns None when insufficient data; dict otherwise
  SC-5: run_mc_for_signal and run_mc_for_portfolio return one result per period
  SC-6: Decimal→percent conversion is applied by pipeline interfaces
"""

import matplotlib
matplotlib.use("Agg")  # force non-interactive backend before any pyplot import

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.monte_carlo import (
    analyze_drawdowns,
    plot_drawdown_distributions,
    run_mc_for_portfolio,
    run_mc_for_signal,
    run_monte_carlo_simulation,
    run_walk_forward_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat_returns(n: int, value: float = 0.1) -> list:
    """Percent-scale returns that are all the same value."""
    return [value] * n


def _synthetic_pct_returns(n: int, seed: int = 0) -> list:
    """Percent-scale returns drawn from a realistic daily distribution."""
    rng = np.random.default_rng(seed)
    return (rng.normal(0.04, 1.0, n)).tolist()


def _dates(n: int) -> list:
    import pandas as pd
    return [d.strftime("%Y-%m-%d") for d in pd.date_range("2020-01-02", periods=n, freq="B")]


# ---------------------------------------------------------------------------
# SC-1: run_monte_carlo_simulation
# ---------------------------------------------------------------------------

class TestRunMonteCarloSimulation:
    def test_output_keys(self):
        returns = _synthetic_pct_returns(200)
        result = run_monte_carlo_simulation(returns, num_simulations=50, simulation_length=50)
        required = {
            "final_returns", "paths", "percentiles",
            "sharpe_ratios", "max_drawdowns",
            "max_drawdown_durations", "total_drawdown_days",
        }
        assert required == set(result.keys())

    def test_paths_shape(self):
        returns = _synthetic_pct_returns(100)
        n_sims, sim_len = 30, 40
        result = run_monte_carlo_simulation(returns, num_simulations=n_sims, simulation_length=sim_len)
        assert result["paths"].shape == (n_sims, sim_len + 1)
        # First column is always 0.0 (starting point)
        np.testing.assert_array_equal(result["paths"][:, 0], 0.0)

    def test_final_returns_shape(self):
        returns = _synthetic_pct_returns(100)
        n_sims = 20
        result = run_monte_carlo_simulation(returns, num_simulations=n_sims, simulation_length=30)
        assert result["final_returns"].shape == (n_sims,)

    def test_percentiles_keys(self):
        returns = _synthetic_pct_returns(100)
        result = run_monte_carlo_simulation(returns, num_simulations=20, simulation_length=30)
        assert set(result["percentiles"].keys()) == {"5", "25", "50", "75", "95"}

    def test_percentile_ordering(self):
        """p5 ≤ p25 ≤ p50 ≤ p75 ≤ p95 at every step."""
        returns = _synthetic_pct_returns(200, seed=42)
        result = run_monte_carlo_simulation(returns, num_simulations=200, simulation_length=50)
        p = result["percentiles"]
        assert np.all(p["5"] <= p["25"])
        assert np.all(p["25"] <= p["50"])
        assert np.all(p["50"] <= p["75"])
        assert np.all(p["75"] <= p["95"])

    def test_max_drawdowns_non_negative(self):
        returns = _synthetic_pct_returns(100)
        result = run_monte_carlo_simulation(returns, num_simulations=20, simulation_length=30)
        assert np.all(result["max_drawdowns"] >= 0)

    def test_default_simulation_length(self):
        """When simulation_length is omitted, it defaults to len(returns)."""
        returns = _synthetic_pct_returns(60)
        result = run_monte_carlo_simulation(returns, num_simulations=10)
        assert result["paths"].shape == (10, len(returns) + 1)


# ---------------------------------------------------------------------------
# SC-2: analyze_drawdowns
# ---------------------------------------------------------------------------

class TestAnalyzeDrawdowns:
    def test_no_drawdown_series(self, tmp_path):
        """Monotonically increasing series → no drawdown periods."""
        returns = list(range(20))  # 0,1,2,...19 — always rising
        result = analyze_drawdowns(
            returns, str(tmp_path), 20, "2020-01-01", "2020-01-31", "test"
        )
        assert result["max_drawdown"] == pytest.approx(0.0)
        assert result["drawdown_periods"] == 0

    def test_known_drawdown(self, tmp_path):
        """
        A series that rises to 10, drops to 5, recovers to 15.
        Max drawdown should be positive and drawdown_periods == 1.
        """
        returns = [0, 2, 4, 6, 8, 10, 8, 6, 5, 7, 10, 12, 15]
        result = analyze_drawdowns(
            returns, str(tmp_path), len(returns),
            "2020-01-01", "2020-01-20", "test"
        )
        assert result["max_drawdown"] > 0
        assert result["drawdown_periods"] >= 1

    def test_required_keys(self, tmp_path):
        returns = [0.0, 1.0, -1.0, 2.0, 0.0, 3.0]
        result = analyze_drawdowns(
            returns, str(tmp_path), 6, "2020-01-01", "2020-01-10", "test"
        )
        required = {
            "max_drawdown", "avg_drawdown", "total_drawdown_days",
            "significant_drawdown_days", "avg_drawdown_length", "avg_calendar_days",
            "drawdown_periods", "significant_periods", "max_drawdown_duration",
            "max_calendar_duration", "drawdown_durations", "calendar_durations",
            "drawdown_magnitudes", "top_significant_periods",
        }
        assert required.issubset(set(result.keys()))

    def test_png_saved(self, tmp_path):
        returns = [0, 1, 2, 1, 3, 2, 4]
        analyze_drawdowns(
            returns, str(tmp_path), 7, "2020-01-01", "2020-01-10", "myport"
        )
        saved = list(tmp_path.glob("myport_drawdown_analysis_7d.png"))
        assert len(saved) == 1

    def test_date_length_mismatch_handled(self, tmp_path):
        """Extra dates should not raise — function truncates to min length."""
        returns = [0.0, 1.0, 2.0]
        dates = ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]  # one extra
        result = analyze_drawdowns(
            returns, str(tmp_path), 3, "2020-01-01", "2020-01-03", "test",
            dates=dates,
        )
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# SC-3: plot_drawdown_distributions (matplotlib mocked)
# ---------------------------------------------------------------------------

class TestPlotDrawdownDistributions:
    def _make_sim_results(self, n: int = 50) -> dict:
        rng = np.random.default_rng(7)
        return {
            "max_drawdowns":          rng.uniform(0, 20, n),
            "max_drawdown_durations": rng.integers(1, 60, n).astype(float),
        }

    def test_output_keys(self, tmp_path):
        sim = self._make_sim_results()
        result = plot_drawdown_distributions(
            sim, actual_max_drawdown=10.0, actual_dd_duration=30,
            period_length=126, output_dir=str(tmp_path), portfolio_name="port",
        )
        required = {
            "dd_mean", "dd_median", "dd_std", "dd_5th", "dd_95th", "dd_percentile",
            "dur_mean", "dur_median", "dur_std", "dur_5th", "dur_95th", "dur_percentile",
        }
        assert required == set(result.keys())

    def test_png_saved(self, tmp_path):
        sim = self._make_sim_results()
        plot_drawdown_distributions(
            sim, 5.0, 20, 63, str(tmp_path), "myport"
        )
        saved = list(tmp_path.glob("myport_drawdown_distributions_63d.png"))
        assert len(saved) == 1

    def test_percentile_in_range(self, tmp_path):
        """dd_percentile must be in [0, 100]."""
        sim = self._make_sim_results(100)
        result = plot_drawdown_distributions(
            sim, 10.0, 40, 126, str(tmp_path), "port"
        )
        assert 0.0 <= result["dd_percentile"] <= 100.0
        assert 0.0 <= result["dur_percentile"] <= 100.0


# ---------------------------------------------------------------------------
# SC-4: run_walk_forward_test
# ---------------------------------------------------------------------------

class TestRunWalkForwardTest:
    def test_returns_none_when_too_short(self, tmp_path):
        returns = _synthetic_pct_returns(30)
        dates = _dates(30)
        result = run_walk_forward_test(dates, returns, test_period_length=30,
                                       output_dir=str(tmp_path), portfolio_name="p")
        assert result is None

    def test_returns_none_insufficient_training(self, tmp_path):
        """test_period_length so large that train < 30 days."""
        returns = _synthetic_pct_returns(40)
        dates = _dates(40)
        result = run_walk_forward_test(dates, returns, test_period_length=15,
                                       output_dir=str(tmp_path), portfolio_name="p")
        # 40 - 15 = 25 training days < 30 minimum
        assert result is None

    def test_returns_dict_when_enough_data(self, tmp_path):
        returns = _synthetic_pct_returns(150)
        dates = _dates(150)
        result = run_walk_forward_test(dates, returns, test_period_length=50,
                                       output_dir=str(tmp_path), portfolio_name="p")
        assert isinstance(result, dict)

    def test_required_keys_present(self, tmp_path):
        returns = _synthetic_pct_returns(150)
        dates = _dates(150)
        result = run_walk_forward_test(dates, returns, test_period_length=50,
                                       output_dir=str(tmp_path), portfolio_name="p")
        required = {
            "period_length", "test_start_date", "test_end_date",
            "actual_final_return", "actual_max_drawdown",
            "actual_dd_duration_trading", "actual_dd_duration_calendar",
            "actual_percentile", "median_forecast", "forecast_error", "percent_error",
            "in_90_interval", "in_50_interval",
        }
        assert required.issubset(set(result.keys()))

    def test_period_length_recorded(self, tmp_path):
        returns = _synthetic_pct_returns(150)
        dates = _dates(150)
        result = run_walk_forward_test(dates, returns, test_period_length=50,
                                       output_dir=str(tmp_path), portfolio_name="p")
        assert result["period_length"] == 50

    def test_walk_forward_png_saved(self, tmp_path):
        returns = _synthetic_pct_returns(150)
        dates = _dates(150)
        run_walk_forward_test(dates, returns, test_period_length=50,
                              output_dir=str(tmp_path), portfolio_name="myport")
        saved = list(tmp_path.glob("myport_walk_forward_50d.png"))
        assert len(saved) == 1


# ---------------------------------------------------------------------------
# SC-5 & SC-6: Pipeline interfaces
# ---------------------------------------------------------------------------

class TestRunMcForSignal:
    def test_returns_one_result_per_period(self, tmp_path):
        n = 200
        rng = np.random.default_rng(42)
        signal_col = rng.random(n) > 0.5
        target_returns = rng.normal(0.0003, 0.01, n)
        bil_returns = np.full(n, 0.0001)
        dates = _dates(n)

        results = run_mc_for_signal(
            signal_col, target_returns, bil_returns, dates,
            output_dir=str(tmp_path), portfolio_name="sig",
            test_period_lengths=[50],
        )
        assert len(results) == 1

    def test_none_entry_for_short_period(self, tmp_path):
        """If the dataset is too short for a requested period, that entry is None."""
        n = 40
        rng = np.random.default_rng(0)
        signal_col = np.ones(n, dtype=bool)
        target_returns = rng.normal(0.0003, 0.01, n)
        bil_returns = np.zeros(n)
        dates = _dates(n)

        results = run_mc_for_signal(
            signal_col, target_returns, bil_returns, dates,
            output_dir=str(tmp_path), portfolio_name="sig",
            test_period_lengths=[40],  # leaves 0 training days
        )
        assert results[0] is None

    def test_decimal_to_pct_conversion(self, tmp_path):
        """
        1% daily return in decimal = 0.01 → should be converted to 1.0 pct-scale.
        The signal is always-on, so daily_pnl = target_returns.
        With 0.01 decimal → 1.0 pct per day over 100 days, the cumulative return
        should be clearly positive (not tiny as if still in decimal scale).
        """
        n = 150
        signal_col = np.ones(n, dtype=bool)
        target_returns = np.full(n, 0.01)   # 1% per day in decimal
        bil_returns = np.zeros(n)
        dates = _dates(n)

        results = run_mc_for_signal(
            signal_col, target_returns, bil_returns, dates,
            output_dir=str(tmp_path), portfolio_name="sig",
            test_period_lengths=[50],
        )
        result = results[0]
        assert result is not None
        # With 1% per day compounded over 50 days: (1.01^50 - 1)*100 ≈ 64.5%
        # If decimal→pct conversion were missing, we'd get ≈ 0.64%
        assert result["actual_final_return"] > 10.0, (
            f"Expected ~64% final return but got {result['actual_final_return']:.2f}% "
            "— decimal→percent conversion may be missing"
        )


class TestRunMcForPortfolio:
    def test_returns_one_result_per_period(self, tmp_path):
        n = 200
        rng = np.random.default_rng(5)
        portfolio_returns = rng.normal(0.0003, 0.01, n)
        dates = _dates(n)

        results = run_mc_for_portfolio(
            portfolio_returns, dates,
            output_dir=str(tmp_path), portfolio_name="port",
            test_period_lengths=[50, 80],
        )
        assert len(results) == 2

    def test_decimal_to_pct_conversion(self, tmp_path):
        """Same conversion check as run_mc_for_signal."""
        n = 150
        portfolio_returns = np.full(n, 0.01)  # 1% per day decimal
        dates = _dates(n)

        results = run_mc_for_portfolio(
            portfolio_returns, dates,
            output_dir=str(tmp_path), portfolio_name="port",
            test_period_lengths=[50],
        )
        result = results[0]
        assert result is not None
        assert result["actual_final_return"] > 10.0, (
            f"Expected ~64% but got {result['actual_final_return']:.2f}% "
            "— decimal→percent conversion may be missing"
        )

    def test_default_periods(self, tmp_path):
        """Default test_period_lengths=[63,126,252]; dataset must be large enough."""
        n = 500
        rng = np.random.default_rng(9)
        portfolio_returns = rng.normal(0.0003, 0.01, n)
        dates = _dates(n)

        results = run_mc_for_portfolio(
            portfolio_returns, dates,
            output_dir=str(tmp_path), portfolio_name="port",
        )
        assert len(results) == 3  # three default periods
