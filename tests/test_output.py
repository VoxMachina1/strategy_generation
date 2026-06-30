"""
Tests for src/output.py — Phase 9: Output and Report Generation.

Covers success criteria:
  SC-1: write_csvs creates all_signals.csv, top_n_signals.csv, and optionally
        all_combos.csv and rsi_search.csv
  SC-2: write_symphony_json creates a valid JSON file round-trippable with json.load
  SC-3: write_report_html creates a self-contained HTML file that:
        - exists and is non-empty
        - contains a sortable table (sortTable JS function present)
        - contains the portfolio equity curve SVG
  SC-4: write_output creates a timestamped run_dir with all artifacts
  SC-5: report.html contains required structural markers for the SC-3 browser check
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.output import (
    write_csvs,
    write_output,
    write_report_html,
    write_symphony_json,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _top_n_df(n_rows=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        rows.append({
            "signal_name":      f"RSI_10_SPY_LT_{30 + i}",
            "target":           "TQQQ",
            "Sharpe_p50":       float(rng.normal(0.8, 0.3)),
            "Sharpe_p10":       float(rng.normal(0.2, 0.3)),
            "Sharpe_p90":       float(rng.normal(1.4, 0.3)),
            "Sharpe_IQR":       float(abs(rng.normal(0.5, 0.1))),
            "Sharpe_Stripped":  float(rng.normal(0.7, 0.3)),
            "Return_p50":       float(rng.normal(0.02, 0.01)),
            "Return_p10":       float(rng.normal(0.01, 0.01)),
            "MaxDD_p90":        float(abs(rng.normal(0.15, 0.05))),
            "Sortino_p50":      float(rng.normal(1.0, 0.3)),
            "Calmar_p50":       float(rng.normal(0.5, 0.2)),
            "Consistency_Score": float(rng.uniform(0.5, 1.0)),
            "N_Iterations":     5,
            "Tail_Concentration": float(rng.uniform(0.1, 0.8)),
        })
    return pd.DataFrame(rows)


def _all_signals_df(n_rows=20, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_rows):
        rows.append({
            "signal_name": f"SIG_{i}",
            "target":      "TQQQ",
            "sharpe":      float(rng.normal(0.5, 0.5)),
            "total_return": float(rng.normal(0.05, 0.1)),
        })
    return pd.DataFrame(rows)


def _mock_returns(n=300, seed=42):
    rng = np.random.default_rng(seed)
    target = rng.normal(0.0003, 0.01, n)
    bil    = np.full(n, 0.0001)
    dates  = pd.date_range("2020-01-02", periods=n, freq="B").to_numpy()
    return target, bil, dates


def _mock_signal_matrix(n=300, n_sigs=5, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random((n, n_sigs)) > 0.5).astype(bool)


def _symphony():
    return {
        "step": "root",
        "rebalance": "daily",
        "children": [{"step": "wt-cash-equal", "children": []}],
    }


# ---------------------------------------------------------------------------
# SC-1: write_csvs
# ---------------------------------------------------------------------------

class TestWriteCsvs:
    def test_all_signals_csv_created(self, tmp_path):
        paths = write_csvs(tmp_path, _all_signals_df(), None, _top_n_df())
        assert Path(paths["all_signals"]).exists()

    def test_top_n_csv_created(self, tmp_path):
        paths = write_csvs(tmp_path, _all_signals_df(), None, _top_n_df())
        assert Path(paths["top_n_signals"]).exists()

    def test_combos_csv_written_when_provided(self, tmp_path):
        combos_df = _all_signals_df(n_rows=10)
        paths = write_csvs(tmp_path, _all_signals_df(), combos_df, _top_n_df())
        assert "all_combos" in paths
        assert Path(paths["all_combos"]).exists()

    def test_combos_csv_skipped_when_none(self, tmp_path):
        paths = write_csvs(tmp_path, _all_signals_df(), None, _top_n_df())
        assert "all_combos" not in paths

    def test_rsi_search_csv_written_when_provided(self, tmp_path):
        rsi_df = pd.DataFrame([{"Signal": "RSI_10_SPY_LT_30", "Sharpe": 1.2}])
        paths = write_csvs(tmp_path, _all_signals_df(), None, _top_n_df(), rsi_df)
        assert "rsi_search" in paths
        assert Path(paths["rsi_search"]).exists()

    def test_csv_contents_round_trip(self, tmp_path):
        top = _top_n_df(3)
        paths = write_csvs(tmp_path, _all_signals_df(), None, top)
        loaded = pd.read_csv(paths["top_n_signals"])
        assert list(loaded.columns) == list(top.columns)
        assert len(loaded) == len(top)

    def test_creates_output_dir_if_absent(self, tmp_path):
        new_dir = tmp_path / "nested" / "dir"
        assert not new_dir.exists()
        write_csvs(new_dir, _all_signals_df(), None, _top_n_df())
        assert new_dir.exists()


# ---------------------------------------------------------------------------
# SC-2: write_symphony_json
# ---------------------------------------------------------------------------

class TestWriteSymphonyJson:
    def test_file_created(self, tmp_path):
        p = write_symphony_json(tmp_path, _symphony())
        assert Path(p).exists()
        assert Path(p).name == "symphony.json"

    def test_valid_json(self, tmp_path):
        p = write_symphony_json(tmp_path, _symphony())
        with open(p) as f:
            data = json.load(f)
        assert data["step"] == "root"

    def test_round_trip_fidelity(self, tmp_path):
        original = {
            "step": "root", "id": "abc", "name": "Test",
            "description": "", "rebalance": "daily",
            "rebalance-corridor-width": 0.0,
            "children": [{"step": "wt-cash-equal", "id": "def", "children": [
                {"step": "if", "id": "ghi", "children": [
                    {"step": "if-child", "id": "jkl", "is-else-condition?": False,
                     "lhs-fn": "relative-strength-index", "lhs-val": "SPY",
                     "lhs-fn-params": {"window": 10}, "comparator": "lt",
                     "rhs-fixed-value?": True, "rhs-val": "30",
                     "condition": {"condition-type": "binary", "comparator": "lt",
                                   "lhs": {"fn": "relative-strength-index", "ticker": "SPY",
                                           "params": {"window": 10}},
                                   "rhs": {"constant": 30.0}},
                     "children": [{"step": "asset", "id": "mno", "ticker": "TQQQ"}]},
                    {"step": "if-child", "id": "pqr", "is-else-condition?": True,
                     "children": [{"step": "asset", "id": "stu", "ticker": "BIL"}]},
                ]},
            ]}],
        }
        p = write_symphony_json(tmp_path, original)
        with open(p) as f:
            loaded = json.load(f)
        assert loaded == original


# ---------------------------------------------------------------------------
# SC-3: write_report_html
# ---------------------------------------------------------------------------

class TestWriteReportHtml:
    def _write(self, tmp_path, top_n=None, n=200):
        target, bil, dates = _mock_returns(n)
        sm = _mock_signal_matrix(n, n_sigs=5)
        signal_names = [f"RSI_10_SPY_LT_{30+i}" for i in range(5)]
        cfg = {"signal_tickers": ["SPY"], "target_tickers": ["TQQQ"]}
        top = top_n if top_n is not None else _top_n_df(5)
        return write_report_html(tmp_path, cfg, top, sm, signal_names, target, bil, dates)

    def test_file_created(self, tmp_path):
        p = self._write(tmp_path)
        assert Path(p).exists()
        assert Path(p).name == "report.html"

    def test_non_empty(self, tmp_path):
        p = self._write(tmp_path)
        assert Path(p).stat().st_size > 1000

    def test_sortable_table_js_present(self, tmp_path):
        p = self._write(tmp_path)
        content = Path(p).read_text(encoding="utf-8")
        assert "sortTable" in content, "sortTable JS function not found in report.html"

    def test_equity_curve_svg_present(self, tmp_path):
        p = self._write(tmp_path)
        content = Path(p).read_text(encoding="utf-8")
        assert "<svg" in content, "SVG equity curve not found in report.html"
        assert "polyline" in content or "Portfolio" in content

    def test_html_structure(self, tmp_path):
        p = self._write(tmp_path)
        content = Path(p).read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "<table" in content
        assert "Tail" in content  # tail section

    def test_empty_top_n_does_not_crash(self, tmp_path):
        p = self._write(tmp_path, top_n=pd.DataFrame())
        assert Path(p).exists()

    def test_tail_concentration_badge_present(self, tmp_path):
        p = self._write(tmp_path)
        content = Path(p).read_text(encoding="utf-8")
        # Badge spans with HIGH/MED/LOW should appear for any Tail_Concentration values
        assert any(label in content for label in ["HIGH", "MED", "LOW"])


# ---------------------------------------------------------------------------
# SC-4 & SC-5: write_output
# ---------------------------------------------------------------------------

class TestWriteOutput:
    def test_creates_timestamped_directory(self, tmp_path):
        target, bil, dates = _mock_returns(200)
        sm = _mock_signal_matrix(200, 3)
        names = [f"RSI_10_SPY_LT_{30+i}" for i in range(3)]

        paths = write_output(
            base_output_dir=tmp_path,
            run_config={"signal_tickers": ["SPY"]},
            all_signals_df=_all_signals_df(),
            top_n_df=_top_n_df(3),
            symphony_dict=_symphony(),
            signal_matrix=sm,
            signal_names=names,
            target_returns_moc=target,
            bil_returns=bil,
            dates=dates,
            run_timestamp="20260101_120000",
        )

        run_dir = tmp_path / "20260101_120000"
        assert run_dir.is_dir()

    def test_all_core_artifacts_present(self, tmp_path):
        target, bil, dates = _mock_returns(200)
        sm = _mock_signal_matrix(200, 3)
        names = [f"RSI_10_SPY_LT_{30+i}" for i in range(3)]

        paths = write_output(
            base_output_dir=tmp_path,
            run_config={},
            all_signals_df=_all_signals_df(),
            top_n_df=_top_n_df(3),
            symphony_dict=_symphony(),
            signal_matrix=sm,
            signal_names=names,
            target_returns_moc=target,
            bil_returns=bil,
            dates=dates,
            run_timestamp="20260101_130000",
        )

        assert "all_signals" in paths
        assert "top_n_signals" in paths
        assert "symphony_json" in paths
        assert "report_html" in paths

        for name, p in paths.items():
            assert Path(p).exists(), f"Missing artifact: {name} at {p}"
