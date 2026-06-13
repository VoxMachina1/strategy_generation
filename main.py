"""
Composer Signal Pipeline — full pipeline entrypoint.

Usage:
    python main.py [options]

Runs all stages from config loading through data fetch, signal generation,
backtesting, validation, tail analysis, Composer export, and output writing.

See .planning/SPEC-remaining.md for the full stage description.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults — merged with config.py values then CLI overrides
# ---------------------------------------------------------------------------

_PIPELINE_DEFAULTS = {
    "validation": {
        "window_type": "walk_forward",
        "train_size":  756,
        "test_size":   63,
    },
    "top_n":            20,
    "run_combos":       True,
    "run_mc":           True,
    "combo_batch_size": 500,
    "top_k_for_combos": 50,
}


# ---------------------------------------------------------------------------
# Progress printer
# ---------------------------------------------------------------------------

class _Progress:
    def __init__(self, total: int):
        self._total = total
        self._current = 0
        self._t0 = None

    def start(self, label: str):
        self._current += 1
        self._t0 = time.time()
        print(f"[{self._current}/{self._total}] {label}...", end="", flush=True)

    def done(self):
        elapsed = time.time() - self._t0
        print(f"  done ({elapsed:.1f}s)")


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------

def _load_config(config_path: str | None) -> dict:
    """Merge pipeline defaults → config.py → optional JSON file."""
    from config import RSI_SEARCH_CONFIG

    cfg = {}
    cfg.update(_PIPELINE_DEFAULTS)
    cfg.update(RSI_SEARCH_CONFIG)

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
        cfg.update(overrides)

    return cfg


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.workers is not None:
        cfg["n_workers"] = args.workers
    if args.window_type is not None:
        cfg.setdefault("validation", {})["window_type"] = args.window_type
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.no_combos:
        cfg["run_combos"] = False
    if args.no_mc:
        cfg["run_mc"] = False
    return cfg


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _stage_fetch_data(cfg: dict, prog: _Progress) -> tuple:
    """Returns (price_df, dates)."""
    from src.data.loader import load_api_keys, check_freshness_and_update
    from src.data.alignment import load_multi_ticker_aligned

    prog.start("Fetching / refreshing price data")

    try:
        api_keys = load_api_keys()
    except ValueError as e:
        print()
        print(f"\nError: {e}")
        sys.exit(1)

    all_tickers = list(dict.fromkeys(
        cfg["signal_tickers"] + cfg["target_tickers"] + [cfg["benchmark_ticker"]]
    ))
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)

    check_freshness_and_update(all_tickers, api_keys, data_dir)
    price_df = load_multi_ticker_aligned(all_tickers, data_dir)
    dates = price_df.index.to_numpy()

    prog.done()
    return price_df, dates


def _stage_build_indicator_cache(cfg: dict, price_df: pd.DataFrame, prog: _Progress) -> dict:
    from src.data.cache import build_indicator_cache
    from src.signals import generate_signal_specs, derive_required_indicators

    prog.start("Building indicator cache")
    specs = generate_signal_specs(cfg)
    required = derive_required_indicators(specs)
    cache = build_indicator_cache(price_df, required)
    prog.done()
    return cache


def _stage_generate_signal_matrix(
    cfg: dict, price_df: pd.DataFrame, indicator_cache: dict, prog: _Progress
) -> tuple:
    from src.signals import generate_signal_matrix

    prog.start("Generating signal matrix")
    signal_matrix, signal_names, signal_metadata, date_index = generate_signal_matrix(
        cfg, price_df, indicator_cache
    )
    if signal_matrix.shape[1] == 0:
        print()
        print("\nError: signal matrix is empty — no signals generated. Check config.")
        sys.exit(1)
    prog.done()
    print(f"       {signal_matrix.shape[1]} signals × {signal_matrix.shape[0]} days")
    return signal_matrix, signal_names, signal_metadata, date_index


def _stage_compute_returns(
    cfg: dict, price_df: pd.DataFrame, prog: _Progress
) -> tuple:
    """Returns (target_returns_dict, bil_returns)."""
    from src.backtest import prepare_moc_returns

    prog.start("Computing MOC returns")
    target_returns_dict = {}
    for ticker in cfg["target_tickers"]:
        raw = price_df[ticker].pct_change().to_numpy()
        target_returns_dict[ticker] = prepare_moc_returns(raw)

    bench = cfg["benchmark_ticker"]
    bil_raw = price_df[bench].pct_change().to_numpy()
    bil_returns = prepare_moc_returns(bil_raw)

    prog.done()
    return target_returns_dict, bil_returns


def _stage_backtest_is(
    cfg: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_dict: dict,
    bil_returns: np.ndarray,
    date_index: np.ndarray,
    prog: _Progress,
) -> pd.DataFrame:
    from src.backtest import batch_backtest

    prog.start("In-sample backtest (all targets)")
    rows = []
    for ticker, tr_moc in target_returns_dict.items():
        df = batch_backtest(signal_matrix, tr_moc, bil_returns, date_index,
                            signal_names=signal_names,
                            n_workers=cfg.get("n_workers"))
        df["target"] = ticker
        rows.append(df)
    is_results_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    prog.done()
    return is_results_df


def _stage_run_combos(
    cfg: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    signal_metadata: list,
    target_returns_dict: dict,
    bil_returns: np.ndarray,
    date_index: np.ndarray,
    prog: _Progress,
) -> pd.DataFrame:
    from src.combos import run_combo_backtests

    prog.start("Generating and backtesting combos")
    combo_rows = run_combo_backtests(
        signal_matrix, signal_names, signal_metadata,
        target_returns_dict, bil_returns, date_index,
        top_k_for_combos=cfg.get("top_k_for_combos", 50),
        config={"combo_batch_size": cfg.get("combo_batch_size", 500)},
    )
    combo_df = pd.DataFrame(combo_rows) if combo_rows else pd.DataFrame()
    prog.done()
    if not combo_df.empty:
        print(f"       {len(combo_df)} combo results")
    return combo_df


def _stage_run_validation(
    cfg: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    signal_metadata: list,
    price_df: pd.DataFrame,
    bil_returns: np.ndarray,
    prog: _Progress,
) -> pd.DataFrame:
    from src.validation import run_validation

    val_cfg = cfg.get("validation", _PIPELINE_DEFAULTS["validation"])
    window_type = val_cfg.get("window_type", "walk_forward")
    window_config = {
        k: v for k, v in val_cfg.items() if k != "window_type"
    }

    prog.start(f"OOS validation ({window_type})")
    oos_raw_df = run_validation(
        signal_matrix, signal_names, signal_metadata,
        price_df, cfg["target_tickers"], bil_returns,
        window_type=window_type,
        window_config=window_config,
        n_workers=cfg.get("n_workers"),
    )
    prog.done()
    return oos_raw_df


def _stage_aggregate_oos(oos_raw_df: pd.DataFrame, prog: _Progress) -> pd.DataFrame:
    from src.validation import aggregate_oos_results

    prog.start("Aggregating OOS results")
    all_signals_df = aggregate_oos_results(oos_raw_df)
    prog.done()
    print(f"       {len(all_signals_df)} (signal, target) pairs")
    return all_signals_df


def _stage_tail_metrics(
    all_signals_df: pd.DataFrame,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_dict: dict,
    prog: _Progress,
) -> pd.DataFrame:
    from src.metrics import tail_metrics

    prog.start("Computing tail metrics")

    _rename = {
        "tail_concentration": "Tail_Concentration",
        "excess_kurtosis":    "Excess_Kurtosis",
        "base_win_rate":      "Base_Win_Rate",
        "stripped_win_rate":  "Stripped_Win_Rate",
        "wr_delta":           "WR_Delta",
        "tail_score":         "Tail_Score",
    }

    sig_idx = {name: i for i, name in enumerate(signal_names)}
    tail_rows = []

    for _, row in all_signals_df.iterrows():
        sig_name = row["signal_name"]
        target   = row["target"]
        entry    = {"signal_name": sig_name, "target": target}

        col_i = sig_idx.get(sig_name)
        tr_moc = target_returns_dict.get(target)

        if col_i is None or tr_moc is None:
            tail_rows.append(entry)
            continue

        # Masked return series: days signal is active get target return, else 0
        signal_col = signal_matrix[:, col_i]
        r = np.where(signal_col, tr_moc, 0.0)

        tm = tail_metrics(r)
        entry.update({_rename[k]: v for k, v in tm.items() if k in _rename})
        tail_rows.append(entry)

    tail_df = pd.DataFrame(tail_rows)
    merged = all_signals_df.merge(tail_df, on=["signal_name", "target"], how="left")
    prog.done()
    return merged


def _stage_select_top_n(
    all_signals_df: pd.DataFrame, top_n: int, prog: _Progress
) -> tuple:
    prog.start(f"Selecting top {top_n} signals")
    sort_col = "Sharpe_p50" if "Sharpe_p50" in all_signals_df.columns else all_signals_df.columns[2]
    top_n_df = all_signals_df.sort_values(sort_col, ascending=False).head(top_n)
    top_n_specs = [
        {"signal_name": r.signal_name, "target_ticker": r.target}
        for r in top_n_df.itertuples()
    ]
    prog.done()
    return top_n_df, top_n_specs


def _stage_build_symphony(top_n_specs: list, prog: _Progress) -> dict:
    from src.composer import build_symphony

    prog.start("Building Composer symphony")
    symphony = build_symphony(top_n_specs)
    prog.done()
    return symphony


def _stage_monte_carlo(
    cfg: dict,
    top_n_df: pd.DataFrame,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_dict: dict,
    bil_returns: np.ndarray,
    dates: np.ndarray,
    output_dir: Path,
    prog: _Progress,
) -> list:
    from src.monte_carlo import run_mc_for_signal

    prog.start("Monte Carlo simulation")
    mc_results = []
    sig_idx = {name: i for i, name in enumerate(signal_names)}

    for _, row in top_n_df.iterrows():
        sig_name = row["signal_name"]
        target   = row["target"]
        col_i    = sig_idx.get(sig_name)
        if col_i is None or target not in target_returns_dict:
            continue
        signal_col = signal_matrix[:, col_i]
        tr_moc     = target_returns_dict[target]
        results = run_mc_for_signal(
            signal_col, tr_moc, bil_returns, dates,
            output_dir=str(output_dir),
            portfolio_name=sig_name.replace("/", "_"),
        )
        mc_results.extend(results)

    prog.done()
    return mc_results


def _stage_write_output(
    cfg: dict,
    all_signals_df: pd.DataFrame,
    top_n_df: pd.DataFrame,
    symphony: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_dict: dict,
    bil_returns: np.ndarray,
    dates: np.ndarray,
    base_output_dir: Path,
    combo_df: pd.DataFrame | None,
    prog: _Progress,
) -> dict:
    from src.output import write_output

    prog.start("Writing output artifacts")

    # Use the first target ticker's returns as the portfolio curve proxy
    first_target = cfg["target_tickers"][0]
    target_returns_moc = target_returns_dict.get(first_target, bil_returns)

    paths = write_output(
        base_output_dir=base_output_dir,
        run_config=cfg,
        all_signals_df=all_signals_df,
        top_n_df=top_n_df,
        symphony_dict=symphony,
        signal_matrix=signal_matrix,
        signal_names=signal_names,
        target_returns_moc=target_returns_moc,
        bil_returns=bil_returns,
        dates=dates,
        all_combos_df=combo_df if (combo_df is not None and not combo_df.empty) else None,
    )
    prog.done()
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Composer Signal Pipeline — run the full discovery pipeline"
    )
    p.add_argument("--config",       default=None,   help="Path to JSON config file")
    p.add_argument("--output",       default="output", help="Base output directory")
    p.add_argument("--workers",      type=int, default=None, help="Process pool size")
    p.add_argument("--window-type",  choices=["walk_forward", "expanding", "rolling"],
                   default=None, dest="window_type")
    p.add_argument("--top-n",        type=int, default=None, dest="top_n")
    p.add_argument("--no-combos",    action="store_true", dest="no_combos")
    p.add_argument("--no-mc",        action="store_true", dest="no_mc")
    p.add_argument("--dry-run",      action="store_true", dest="dry_run")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()

    cfg = _load_config(args.config)
    cfg = _apply_cli_overrides(cfg, args)

    if args.dry_run:
        print("Resolved config:")
        print(json.dumps(cfg, indent=2, default=str))
        return

    run_combos = cfg.get("run_combos", True)
    run_mc     = cfg.get("run_mc", True)

    # Count active stages
    n_stages = 11  # base stages always run
    if run_combos:
        n_stages += 1
    if run_mc:
        n_stages += 1

    prog = _Progress(n_stages)
    base_output_dir = Path(args.output)

    try:
        price_df, dates = _stage_fetch_data(cfg, prog)

        indicator_cache = _stage_build_indicator_cache(cfg, price_df, prog)

        signal_matrix, signal_names, signal_metadata, date_index = \
            _stage_generate_signal_matrix(cfg, price_df, indicator_cache, prog)

        target_returns_dict, bil_returns = _stage_compute_returns(cfg, price_df, prog)

        _stage_backtest_is(cfg, signal_matrix, signal_names,
                           target_returns_dict, bil_returns, date_index, prog)

        combo_df = pd.DataFrame()
        if run_combos:
            combo_df = _stage_run_combos(
                cfg, signal_matrix, signal_names, signal_metadata,
                target_returns_dict, bil_returns, date_index, prog
            )

        oos_raw_df = _stage_run_validation(
            cfg, signal_matrix, signal_names, signal_metadata,
            price_df, bil_returns, prog
        )

        all_signals_df = _stage_aggregate_oos(oos_raw_df, prog)

        all_signals_df = _stage_tail_metrics(
            all_signals_df, signal_matrix, signal_names, target_returns_dict, prog
        )

        top_n_df, top_n_specs = _stage_select_top_n(
            all_signals_df, cfg.get("top_n", 20), prog
        )

        symphony = _stage_build_symphony(top_n_specs, prog)

        if run_mc:
            _stage_monte_carlo(
                cfg, top_n_df, signal_matrix, signal_names,
                target_returns_dict, bil_returns, dates,
                base_output_dir, prog
            )

        paths = _stage_write_output(
            cfg, all_signals_df, top_n_df, symphony,
            signal_matrix, signal_names, target_returns_dict, bil_returns, dates,
            base_output_dir,
            combo_df if run_combos else None,
            prog,
        )

        print("\n=== Pipeline complete ===")
        for name, path in paths.items():
            print(f"  {name}: {path}")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception:
        print("\nPipeline failed:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
