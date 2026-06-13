"""
Integration smoke test for the full pipeline (no network calls).

Exercises all pipeline stages end-to-end on synthetic data by patching out the
data-fetch layer. Catches stage-wiring bugs (wrong argument order, missing
return values, etc.) that unit tests can't detect.

Does NOT assert specific numeric values — only that:
  - the pipeline runs to completion without raising
  - output artifacts exist and are non-empty
  - key columns are present in the output CSVs
  - the symphony JSON is structurally valid
"""

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

_N_DAYS = 500
_SIGNAL_TICKERS = ["SPY"]
_TARGET_TICKERS = ["TQQQ", "BIL"]
_BENCHMARK = "BIL"
_ALL_TICKERS = ["SPY", "TQQQ", "BIL"]

_RNG = np.random.default_rng(42)


def _synthetic_price_df() -> pd.DataFrame:
    dates = pd.date_range("2019-01-02", periods=_N_DAYS, freq="B")
    data = {}
    for ticker in _ALL_TICKERS:
        drift = 0.0003 if ticker != "BIL" else 0.00005
        vol   = 0.01   if ticker != "BIL" else 0.0001
        log_ret = _RNG.normal(drift, vol, _N_DAYS)
        data[ticker] = 100.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame(data, index=dates)


_PRICE_DF = _synthetic_price_df()


# ---------------------------------------------------------------------------
# Minimal config (tiny windows so the test is fast)
# ---------------------------------------------------------------------------

_CFG = {
    "signal_tickers":       _SIGNAL_TICKERS,
    "target_tickers":       _TARGET_TICKERS,
    "benchmark_ticker":     _BENCHMARK,
    "rsi_windows":          [10],
    "rsi_thresholds":       [30, 70],
    "comparators":          ["lt", "gt"],   # config.py key
    "rsi_comparators":      ["lt", "gt"],   # signals.py key (both present for direct stage calls)
    "validation": {
        "window_type": "walk_forward",
        "train_size":  150,
        "test_size":   50,
    },
    "top_n":            3,
    "run_combos":       False,   # keep test fast
    "run_mc":           False,   # keep test fast
    "combo_batch_size": 100,
    "top_k_for_combos": 10,
    "min_trades":       1,
    "min_win_rate":     0.0,
    "filter_benchmark_negative": False,
}


# ---------------------------------------------------------------------------
# Patch helpers — replace network I/O with no-ops / synthetic data
# ---------------------------------------------------------------------------

def _noop(*args, **kwargs):
    pass


def _fake_load_api_keys():
    return ["fake_key"]


def _fake_price_df(*args, **kwargs):
    return _PRICE_DF


def _fake_dates(*args, **kwargs):
    return _PRICE_DF, _PRICE_DF.index.to_numpy()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

class TestPipelineIntegration:
    """Run every pipeline stage on synthetic data and assert structural correctness."""

    @pytest.fixture()
    def run_output(self, tmp_path):
        """
        Execute the pipeline (stages 2-15, minus fetch/network) and return
        the artifacts dict from write_output.
        """
        import main as pipeline

        prog = pipeline._Progress(13)

        # Inject synthetic price_df directly — skip network fetch
        price_df = _PRICE_DF.copy()
        dates    = price_df.index.to_numpy()

        indicator_cache = pipeline._stage_build_indicator_cache(_CFG, price_df, prog)

        signal_matrix, signal_names, signal_metadata, date_index = \
            pipeline._stage_generate_signal_matrix(_CFG, price_df, indicator_cache, prog)

        target_returns_dict, bil_returns = \
            pipeline._stage_compute_returns(_CFG, price_df, prog)

        pipeline._stage_backtest_is(
            _CFG, signal_matrix, signal_names,
            target_returns_dict, bil_returns, date_index, prog
        )

        oos_raw_df = pipeline._stage_run_validation(
            _CFG, signal_matrix, signal_names, signal_metadata,
            price_df, bil_returns, prog
        )

        all_signals_df = pipeline._stage_aggregate_oos(oos_raw_df, prog)

        all_signals_df = pipeline._stage_tail_metrics(
            all_signals_df, signal_matrix, signal_names, target_returns_dict, prog
        )

        top_n_df, top_n_specs = pipeline._stage_select_top_n(
            all_signals_df, _CFG["top_n"], _CFG, prog
        )

        symphony = pipeline._stage_build_symphony(top_n_specs, prog)

        paths = pipeline._stage_write_output(
            _CFG, all_signals_df, top_n_df, symphony,
            signal_matrix, signal_names, target_returns_dict, bil_returns, dates,
            tmp_path, None, prog
        )

        return paths, top_n_df, all_signals_df, symphony

    def test_all_artifacts_exist(self, run_output):
        paths, *_ = run_output
        for name, path in paths.items():
            assert Path(path).exists(), f"Missing artifact: {name} → {path}"

    def test_all_signals_csv_has_required_columns(self, run_output):
        paths, _, all_signals_df, _ = run_output
        df = pd.read_csv(paths["all_signals"])
        for col in ("signal_name", "target", "Sharpe_p50", "N_Iterations"):
            assert col in df.columns, f"Missing column: {col}"

    def test_top_n_csv_non_empty(self, run_output):
        paths, *_ = run_output
        df = pd.read_csv(paths["top_n_signals"])
        assert len(df) > 0

    def test_tail_columns_populated(self, run_output):
        paths, *_ = run_output
        df = pd.read_csv(paths["top_n_signals"])
        tail_cols = [c for c in ("Tail_Concentration", "WR_Delta", "Tail_Score")
                     if c in df.columns]
        assert tail_cols, "No tail metric columns found in top_n_signals.csv"
        # At least some rows should have non-NaN tail values
        assert df[tail_cols].notna().any().any(), \
            "All tail metric values are NaN — tail metrics stage is not running"

    def test_symphony_json_valid_structure(self, run_output):
        paths, *_ = run_output
        with open(paths["symphony_json"]) as f:
            sym = json.load(f)
        assert sym["step"] == "root"
        assert sym["children"][0]["step"] == "wt-cash-equal"

    def test_report_html_exists_and_non_empty(self, run_output):
        paths, *_ = run_output
        p = Path(paths["report_html"])
        assert p.exists()
        assert p.stat().st_size > 500

    def test_mode_c_leaf_integration(self, run_output, tmp_path):
        """insert_into_symphony with leaf mode produces a valid extended symphony."""
        import main as pipeline
        from src.composer import insert_into_symphony

        _, top_n_df, _, base_symphony = run_output

        top_n_specs = [
            {"signal_name": r.signal_name, "target_ticker": r.target}
            for r in top_n_df.head(2).itertuples()
        ]
        original_count = len(base_symphony["children"][0]["children"])
        result = insert_into_symphony(base_symphony, top_n_specs, mode="leaf")
        new_count = len(result["children"][0]["children"])
        assert new_count == original_count + len(top_n_specs)

    def test_mode_c_root_integration(self, run_output):
        """insert_into_symphony with root mode produces a valid wrapped symphony."""
        from src.composer import insert_into_symphony

        _, top_n_df, _, base_symphony = run_output
        spec = [{"signal_name": top_n_df.iloc[0]["signal_name"],
                 "target_ticker": top_n_df.iloc[0]["target"]}]

        result = insert_into_symphony(base_symphony, spec, mode="root")
        assert result["step"] == "root"
        outer_if = result["children"][0]["children"][0]
        assert outer_if["step"] == "if"
        # True-child should nest a wt-cash-equal
        inner_wt = outer_if["children"][0]["children"][0]
        assert inner_wt["step"] == "wt-cash-equal"


# ---------------------------------------------------------------------------
# Symphony round-trip verification tests
# ---------------------------------------------------------------------------

class TestVerifySymphony:
    """
    Tests for _stage_verify_symphony and the verify_composer_output integration.

    Verifies that:
      - the stage runs and returns a dict keyed by signal name
      - well-formed signals (built from known RSI conditions) achieve ≥99% match
      - a deliberately broken signal (wrong threshold) triggers warning=True
      - the stage is actually invoked in the pipeline (not silently skipped)
    """

    @pytest.fixture()
    def verify_inputs(self):
        """Build minimal inputs: a known RSI signal, its matrix column, price_df."""
        from src.indicators import calculate_rsi
        from src.composer import build_symphony

        price_df = _PRICE_DF.copy()
        rsi_vals = calculate_rsi(price_df["SPY"], 10).to_numpy(dtype=float)
        signal_col = np.where(np.isnan(rsi_vals), False, rsi_vals < 30).astype(bool)
        signal_matrix = signal_col.reshape(-1, 1)
        signal_names = ["RSI_10_SPY_LT_30"]
        symphony = build_symphony([{"signal_name": "RSI_10_SPY_LT_30",
                                    "target_ticker": "TQQQ"}])
        return symphony, signal_matrix, signal_names, price_df

    def test_returns_dict_keyed_by_signal_name(self, verify_inputs):
        import main as pipeline
        symphony, signal_matrix, signal_names, price_df = verify_inputs
        prog = pipeline._Progress(1)
        results = pipeline._stage_verify_symphony(
            symphony, signal_matrix, signal_names, price_df, prog
        )
        assert isinstance(results, dict)
        assert "RSI_10_SPY_LT_30" in results

    def test_known_signal_achieves_99pct_match(self, verify_inputs):
        import main as pipeline
        symphony, signal_matrix, signal_names, price_df = verify_inputs
        prog = pipeline._Progress(1)
        results = pipeline._stage_verify_symphony(
            symphony, signal_matrix, signal_names, price_df, prog
        )
        r = results["RSI_10_SPY_LT_30"]
        assert r["match_rate"] is not None
        assert r["match_rate"] >= 0.99, f"match_rate={r['match_rate']:.4f}"
        assert r["warning"] is False

    def test_broken_signal_triggers_warning(self):
        """A symphony built with the wrong threshold should fail round-trip."""
        import main as pipeline
        from src.indicators import calculate_rsi
        from src.composer import build_symphony

        price_df = _PRICE_DF.copy()
        rsi_vals = calculate_rsi(price_df["SPY"], 10).to_numpy(dtype=float)
        # Matrix uses threshold 30 ...
        signal_col = np.where(np.isnan(rsi_vals), False, rsi_vals < 30).astype(bool)
        signal_matrix = signal_col.reshape(-1, 1)
        signal_names = ["RSI_10_SPY_LT_30"]
        # ... but symphony encodes threshold 70 — deliberate mismatch
        symphony = build_symphony([{"signal_name": "RSI_10_SPY_LT_70",
                                    "target_ticker": "TQQQ"}])

        prog = pipeline._Progress(1)
        results = pipeline._stage_verify_symphony(
            symphony, signal_matrix, signal_names, price_df, prog
        )
        # Signal not found in symphony → warning=True, match_rate=None
        assert results["RSI_10_SPY_LT_30"]["warning"] is True

    def test_verify_stage_called_in_pipeline(self, tmp_path):
        """verify_results is populated (not empty) after a full pipeline run."""
        import main as pipeline

        prog = pipeline._Progress(13)
        price_df = _PRICE_DF.copy()

        indicator_cache = pipeline._stage_build_indicator_cache(_CFG, price_df, prog)
        signal_matrix, signal_names, signal_metadata, date_index = \
            pipeline._stage_generate_signal_matrix(_CFG, price_df, indicator_cache, prog)
        target_returns_dict, bil_returns = \
            pipeline._stage_compute_returns(_CFG, price_df, prog)
        pipeline._stage_backtest_is(
            _CFG, signal_matrix, signal_names,
            target_returns_dict, bil_returns, date_index, prog
        )
        oos_raw_df = pipeline._stage_run_validation(
            _CFG, signal_matrix, signal_names, signal_metadata,
            price_df, bil_returns, prog
        )
        all_signals_df = pipeline._stage_aggregate_oos(oos_raw_df, prog)
        all_signals_df = pipeline._stage_tail_metrics(
            all_signals_df, signal_matrix, signal_names, target_returns_dict, prog
        )
        top_n_df, top_n_specs = pipeline._stage_select_top_n(
            all_signals_df, _CFG["top_n"], _CFG, prog
        )
        symphony = pipeline._stage_build_symphony(top_n_specs, prog)

        verify_results = pipeline._stage_verify_symphony(
            symphony, signal_matrix, signal_names, price_df, prog
        )

        # verify_composer_output returns one entry per unique signal name
        # (not per (signal, target) pair), so compare against the unique set
        assert len(verify_results) == len(set(signal_names))

        # Every result must have the required keys
        for name, r in verify_results.items():
            assert "match_rate" in r
            assert "warning" in r

        # Signals that ARE in the symphony should have a real match_rate
        symphony_signal_names = {
            spec["signal_name"] for spec in top_n_specs
        }
        for name in symphony_signal_names:
            if name in verify_results:
                r = verify_results[name]
                assert r["match_rate"] is not None, \
                    f"{name} is in symphony but got match_rate=None"
                assert r["match_rate"] >= 0.99, \
                    f"{name} match_rate={r['match_rate']:.4f} below 99%"


# ---------------------------------------------------------------------------
# Quality filter tests
# ---------------------------------------------------------------------------

class TestQualityFilters:
    """Tests for _apply_quality_filters — the pre-ranking signal filter."""

    def _make_df(self, **kwargs):
        """Build a minimal signals DataFrame with one row, overridable per column."""
        import pandas as pd
        defaults = {
            "signal_name":        "RSI_10_SPY_LT_30",
            "target":             "TQQQ",
            "Sharpe_p50":         1.5,
            "N_Iterations":       10,
            "Consistency_Score":  0.70,
            "Stripped_Win_Rate":  0.60,
            "Base_Win_Rate":      0.55,
            "Tail_Concentration": 0.50,
        }
        defaults.update(kwargs)
        return pd.DataFrame([defaults])

    def _default_cfg(self):
        return {
            "min_stripped_win_rate": 0.55,
            "min_base_win_rate":     0.45,
            "max_tail_concentration": 0.80,
            "min_consistency_score": 0.60,
            "min_n_iterations":      5,
        }

    def test_good_signal_passes_all_filters(self):
        import main as pipeline
        df = self._make_df()
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 1

    def test_low_stripped_win_rate_removed(self):
        import main as pipeline
        df = self._make_df(Stripped_Win_Rate=0.40)
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 0

    def test_low_base_win_rate_removed(self):
        import main as pipeline
        df = self._make_df(Base_Win_Rate=0.30)
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 0

    def test_high_tail_concentration_removed(self):
        import main as pipeline
        df = self._make_df(Tail_Concentration=0.95)
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 0

    def test_low_consistency_score_removed(self):
        import main as pipeline
        df = self._make_df(Consistency_Score=0.40)
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 0

    def test_too_few_iterations_removed(self):
        import main as pipeline
        df = self._make_df(N_Iterations=3)
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 0

    def test_missing_column_skipped_gracefully(self):
        """If a tail metric column isn't present, that filter is skipped."""
        import main as pipeline
        df = self._make_df()
        df = df.drop(columns=["Stripped_Win_Rate", "Tail_Concentration"])
        result = pipeline._apply_quality_filters(df, self._default_cfg())
        assert len(result) == 1

    def test_config_thresholds_respected(self):
        """Overriding a threshold in cfg changes which signals pass."""
        import main as pipeline
        df = self._make_df(Stripped_Win_Rate=0.52)
        # Fails at default 0.55
        assert len(pipeline._apply_quality_filters(df, self._default_cfg())) == 0
        # Passes at loosened 0.50
        loose_cfg = {**self._default_cfg(), "min_stripped_win_rate": 0.50}
        assert len(pipeline._apply_quality_filters(df, loose_cfg)) == 1
