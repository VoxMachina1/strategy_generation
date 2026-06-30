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
    from config import PIPELINE_CONFIG

    cfg = dict(PIPELINE_CONFIG)

    if config_path:
        with open(config_path) as f:
            overrides = json.load(f)
        cfg.update(overrides)

    return cfg


def _apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.workers is not None:
        cfg["workers"] = args.workers
    if args.window_type is not None:
        cfg.setdefault("validation", {})["window_type"] = args.window_type
    if args.top_n is not None:
        cfg["top_n"] = args.top_n
    if args.no_combos:
        cfg["run_combos"] = False
    if args.no_mc:
        cfg["run_mc"] = False
    if args.insert_into is not None:
        cfg["insert_into"] = args.insert_into
        cfg["insert_mode"] = args.insert_mode
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

    dl_cfg = cfg.get("dual_layer", {})
    defense_tickers = dl_cfg.get("defensive_target_tickers", []) if dl_cfg.get("enabled") else []
    extra = [cfg["benchmark_ticker"]]
    if cfg.get("safe_asset_ticker"):
        extra.append(cfg["safe_asset_ticker"])
    all_tickers = list(dict.fromkeys(
        cfg["signal_tickers"] + cfg["target_tickers"] + defense_tickers + extra
    ))
    from config import DATA_DIR
    data_dir = DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)

    check_freshness_and_update(all_tickers, api_keys, data_dir)
    price_df = load_multi_ticker_aligned(all_tickers, data_dir)
    dates = price_df.index.to_numpy()

    prog.done()
    return price_df, dates


def _stage_split_holdout(price_df: pd.DataFrame, cfg: dict) -> tuple:
    """
    Split price_df into (train_df, holdout_df) at the configured holdout_cutoff.

    Returns (price_df, None) when dual_layer is disabled — the full dataset is
    used for training and no holdout evaluation is performed.
    """
    dl_cfg = cfg.get("dual_layer", {})
    if not dl_cfg.get("enabled"):
        return price_df, None

    cutoff = pd.Timestamp(dl_cfg["holdout_cutoff"])
    train_df = price_df[price_df.index < cutoff]
    holdout_df = price_df[price_df.index >= cutoff]

    print(f"  Holdout split: {len(train_df)} train days / {len(holdout_df)} holdout days"
          f" (cutoff {dl_cfg['holdout_cutoff']})")
    return train_df, holdout_df


def _normalise_cfg_for_signals(cfg: dict) -> dict:
    """
    generate_signal_specs() uses "rsi_comparators" but config.py exposes "comparators".
    Return a shallow copy with the key aligned.
    """
    out = dict(cfg)
    if "rsi_comparators" not in out and "comparators" in out:
        out["rsi_comparators"] = out["comparators"]
    return out


def _stage_build_indicator_cache(cfg: dict, price_df: pd.DataFrame, prog: _Progress) -> dict:
    from src.data.cache import build_indicator_cache
    from src.signals import generate_signal_specs, derive_required_indicators

    prog.start("Building indicator cache")
    specs = generate_signal_specs(_normalise_cfg_for_signals(cfg))
    required = derive_required_indicators(specs)
    cache = build_indicator_cache(price_df, required)
    prog.done()
    return cache


def _stage_generate_signal_matrix(
    cfg: dict, price_df: pd.DataFrame, indicator_cache: dict, prog: _Progress
) -> tuple:
    from src.signals import generate_signal_matrix, generate_signal_specs

    prog.start("Generating signal matrix")
    specs = generate_signal_specs(_normalise_cfg_for_signals(cfg))

    # Spec breakdown by indicator type
    by_type: dict[str, int] = {}
    for s in specs:
        by_type[s.lhs_fn] = by_type.get(s.lhs_fn, 0) + 1
    breakdown = "  |  ".join(f"{fn}: {n:,}" for fn, n in sorted(by_type.items()))
    print(f"\n       {len(specs):,} specs total  ({breakdown})")

    date_index = price_df.index.to_numpy()
    signal_matrix, signal_names, signal_metadata = generate_signal_matrix(
        specs, indicator_cache, date_index
    )
    if signal_matrix.shape[1] == 0:
        print()
        print("\nError: signal matrix is empty — no signals generated. Check config.")
        sys.exit(1)
    prog.done()
    print(f"       {signal_matrix.shape[1]:,} signals × {signal_matrix.shape[0]:,} days")
    return signal_matrix, signal_names, signal_metadata, date_index


def _stage_compute_returns(
    cfg: dict, price_df: pd.DataFrame, prog: _Progress
) -> tuple:
    """Returns (target_returns_dict, bil_returns).

    bil_returns is the safe-asset return series — what the strategy earns
    when no signal is firing. It comes from safe_asset_ticker (e.g. BIL),
    NOT benchmark_ticker (e.g. SPY). Keeping these separate matters:
    benchmark_ticker is what performance is measured against; safe_asset_ticker
    is what the portfolio actually holds during inactive periods.
    """
    from src.backtest import prepare_moc_returns

    prog.start("Computing MOC returns")
    target_returns_dict = {}
    for ticker in cfg["target_tickers"]:
        raw = price_df[ticker].pct_change().to_numpy()
        target_returns_dict[ticker] = prepare_moc_returns(raw)

    # Use safe_asset_ticker if present; fall back to benchmark_ticker for
    # configs that predate this field.
    safe_asset = cfg.get("safe_asset_ticker") or cfg["benchmark_ticker"]
    bil_raw = price_df[safe_asset].pct_change().to_numpy()
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
        metrics = batch_backtest(signal_matrix, tr_moc, bil_returns)
        n = signal_matrix.shape[1]
        df = pd.DataFrame({k: v for k, v in metrics.items()})
        df["signal_name"] = signal_names[:n]
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
    import math
    from src.combos import run_combo_backtests

    top_k = cfg.get("top_k_for_combos", 50)
    batch_size = cfg.get("combo_batch_size", 500)
    pool_size = min(top_k, signal_matrix.shape[1])
    n_combos = math.comb(pool_size, 2) * 4
    n_batches = (n_combos + batch_size - 1) // batch_size
    n_targets = len(target_returns_dict)

    prog.start(
        f"Combos  (pool={pool_size} signals → {n_combos:,} combos × {n_targets} tickers"
        f" = {n_batches} batches)"
    )

    def _combo_progress(batch_num: int, total_batches: int, n_results: int):
        pct = batch_num / total_batches * 100
        print(
            f"\r       batch {batch_num}/{total_batches} ({pct:.0f}%)  "
            f"{n_results:,} results so far   ",
            end="",
            flush=True,
        )

    combo_rows = run_combo_backtests(
        signal_matrix, signal_names, signal_metadata,
        target_returns_dict, bil_returns, date_index,
        top_k_for_combos=top_k,
        config={"combo_batch_size": batch_size},
        progress_fn=_combo_progress,
    )
    print()  # newline after the \r progress line
    combo_df = pd.DataFrame(combo_rows) if combo_rows else pd.DataFrame()
    prog.done()
    if not combo_df.empty:
        print(f"       {len(combo_df):,} combo results")
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
    from src.validation import (
        run_validation,
        generate_walk_forward_windows,
        generate_expanding_windows,
        generate_rolling_windows,
    )

    val_cfg = cfg.get("validation", {"window_type": "walk_forward", "train_size": 756, "test_size": 63})
    window_type = val_cfg.get("window_type", "walk_forward")
    window_config = {k: v for k, v in val_cfg.items() if k != "window_type"}

    n_days = signal_matrix.shape[0]
    if window_type == "walk_forward":
        n_windows = len(generate_walk_forward_windows(
            n_days, window_config["train_size"], window_config["test_size"]
        ))
    elif window_type == "expanding":
        n_windows = len(generate_expanding_windows(
            n_days, window_config["initial_train"], window_config["test_size"]
        ))
    else:
        n_windows = len(generate_rolling_windows(
            n_days, window_config["train_size"], window_config["test_size"], window_config["step"]
        ))

    target_tickers = cfg["target_tickers"]
    n_targets = len(target_tickers)
    prog.start(
        f"OOS validation ({window_type})"
        f"  {n_windows} windows × {n_targets} tickers = {n_windows * n_targets} backtests"
    )

    _ticker_state: dict = {"current": "", "t0": 0.0}

    def _validation_progress(ticker: str, done: int, total: int):
        import time as _time
        if ticker != _ticker_state["current"]:
            if _ticker_state["current"]:
                elapsed = _time.time() - _ticker_state["t0"]
                print(
                    f"\r       {_ticker_state['current']:8s}  {total}/{total} windows"
                    f"  ({elapsed:.1f}s)              "
                )
            _ticker_state["current"] = ticker
            _ticker_state["t0"] = _time.time()
        pct = done / total * 100
        print(
            f"\r       {ticker:8s}  {done}/{total} windows  ({pct:.0f}%)   ",
            end="",
            flush=True,
        )

    oos_raw_df = run_validation(
        signal_matrix, signal_names, signal_metadata,
        price_df, target_tickers, bil_returns,
        window_type=window_type,
        window_config=window_config,
        n_workers=cfg.get("workers", cfg.get("n_workers")),
        progress_fn=_validation_progress,
    )

    # Print completion line for the last ticker
    if _ticker_state["current"]:
        import time as _time
        elapsed = _time.time() - _ticker_state["t0"]
        print(
            f"\r       {_ticker_state['current']:8s}  {n_windows}/{n_windows} windows"
            f"  ({elapsed:.1f}s)              "
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

    if all_signals_df.empty:
        prog.done()
        return all_signals_df

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

        # Active-day returns only — tail_metrics expects signal days, not the
        # full series. Including inactive days (zeros) destroys win rate stats.
        signal_col = signal_matrix[:, col_i]
        r = tr_moc[signal_col]

        if len(r) == 0:
            tail_rows.append(entry)
            continue

        tm = tail_metrics(r)
        entry.update({_rename[k]: v for k, v in tm.items() if k in _rename})
        tail_rows.append(entry)

    tail_df = pd.DataFrame(tail_rows)
    merged = all_signals_df.merge(tail_df, on=["signal_name", "target"], how="left")
    prog.done()
    return merged


def _apply_quality_filters(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Drop signals that fail minimum quality thresholds before top-N ranking."""
    filters = {
        "Stripped_Win_Rate": (">=", cfg.get("min_stripped_win_rate", 0.55)),
        "Base_Win_Rate":     (">=", cfg.get("min_base_win_rate",     0.45)),
        "Tail_Concentration":("<=", cfg.get("max_tail_concentration", 0.80)),
        "Consistency_Score": (">=", cfg.get("min_consistency_score", 0.60)),
        "N_Iterations":      (">=", cfg.get("min_n_iterations",       5)),
    }
    before = len(df)
    for col, (op, threshold) in filters.items():
        if col not in df.columns:
            continue
        if op == ">=":
            df = df[df[col].fillna(-1) >= threshold]
        elif op == "<=":
            df = df[df[col].fillna(999) <= threshold]
    after = len(df)
    if before != after:
        print(f"  Quality filters removed {before - after} signals ({after} remain)")
    return df


def _stage_select_top_n(
    all_signals_df: pd.DataFrame, top_n: int, cfg: dict, prog: _Progress
) -> tuple:
    prog.start(f"Selecting top {top_n} signals")
    filtered_df = _apply_quality_filters(all_signals_df, cfg)
    sort_col = "Sharpe_p50" if "Sharpe_p50" in filtered_df.columns else filtered_df.columns[2]
    top_n_df = filtered_df.sort_values(sort_col, ascending=False).head(top_n)
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


def _stage_mode_c(
    symphony: dict,
    insert_into: str,
    insert_mode: str,
    top_n_specs: list,
    safe_asset: str,
    prog: _Progress,
) -> dict:
    """Load an existing symphony JSON and insert top-N signals into it."""
    from src.composer import insert_into_symphony

    prog.start(f"Mode C — inserting into {Path(insert_into).name} ({insert_mode})")
    with open(insert_into) as f:
        existing = json.load(f)
    result = insert_into_symphony(existing, top_n_specs, mode=insert_mode,
                                  safe_asset=safe_asset)
    prog.done()
    return result


def _mc_worker(args):
    """Top-level function for process pool — must be importable at module level."""
    from src.monte_carlo import run_mc_for_signal
    signal_col, tr_moc, bil_returns, dates, mc_dir, portfolio_name = args
    return run_mc_for_signal(
        signal_col, tr_moc, bil_returns, dates,
        output_dir=mc_dir,
        portfolio_name=portfolio_name,
    )


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
    from concurrent.futures import ProcessPoolExecutor, as_completed

    mc_dir = str(output_dir / "mc_plots")
    (output_dir / "mc_plots").mkdir(exist_ok=True)

    sig_idx = {name: i for i, name in enumerate(signal_names)}

    tasks = []
    for _, row in top_n_df.iterrows():
        sig_name = row["signal_name"]
        target   = row["target"]
        col_i    = sig_idx.get(sig_name)
        if col_i is None or target not in target_returns_dict:
            continue
        tasks.append((
            signal_matrix[:, col_i],
            target_returns_dict[target],
            bil_returns,
            dates,
            mc_dir,
            sig_name.replace("/", "_"),
        ))

    n_tasks = len(tasks)
    n_workers = cfg.get("workers", cfg.get("mc_workers", min(4, n_tasks or 1)))
    prog.start(f"Monte Carlo simulation (0/{n_tasks})")

    mc_results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_mc_worker, t): t for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            print(f"\r  [13/14] Monte Carlo simulation ({completed}/{n_tasks})...", end="", flush=True)
            result = fut.result()
            mc_results.extend(result)

    print()  # newline after progress line
    prog.done()
    return mc_results


def _stage_verify_symphony(
    symphony: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    price_df: pd.DataFrame,
    prog: _Progress,
) -> dict:
    """
    Re-evaluate symphony conditions against price_df and compare to signal_matrix.
    Logs a warning for any signal whose round-trip match rate falls below 99%.
    Returns the full verification results dict.
    """
    from src.composer import verify_composer_output

    prog.start("Verifying symphony round-trip")
    results = verify_composer_output(symphony, signal_matrix, signal_names, price_df)

    # Only warn about signals that ARE in the symphony but fail round-trip.
    # match_rate=None means the signal wasn't selected into top-N — expected, not a warning.
    failures = [name for name, r in results.items()
                if r.get("warning") and r.get("match_rate") is not None]
    if failures:
        print(f"\n  WARNING: {len(failures)} symphony signal(s) below 99% match rate:")
        for name in failures:
            rate = results[name]["match_rate"]
            print(f"    {name}: {rate:.1%}")
    else:
        n_verified = sum(1 for r in results.values() if r.get("match_rate") is not None)
        print(f"  {n_verified} symphony signals verified ✓")

    prog.done()
    return results


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
    run_timestamp: str | None = None,
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
        run_timestamp=run_timestamp,
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
    p.add_argument("--insert-into",  default=None, dest="insert_into",
                   metavar="SYMPHONY_JSON",
                   help="Path to an existing symphony JSON; insert top-N signals into it")
    p.add_argument("--insert-mode",  choices=["leaf", "root"], default="leaf",
                   dest="insert_mode",
                   help="leaf: append inside wt-cash-equal; root: wrap existing strategy")
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

    run_combos  = cfg.get("run_combos", True)
    run_mc      = cfg.get("run_mc", True)
    insert_into = cfg.get("insert_into")

    if insert_into and not Path(insert_into).exists():
        print(f"Error: --insert-into path not found: {insert_into}")
        sys.exit(1)

    # Count active stages
    n_stages = 12  # base stages always run (includes verify)
    if run_combos:
        n_stages += 1
    if run_mc:
        n_stages += 1
    if insert_into:
        n_stages += 1

    prog = _Progress(n_stages)
    base_output_dir = Path(args.output)
    from datetime import datetime as _dt
    run_timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        price_df, dates = _stage_fetch_data(cfg, prog)

        # Holdout split — train_df is used for all three passes.
        # holdout_df is reserved for final validation after assembly.
        train_df, holdout_df = _stage_split_holdout(price_df, cfg)

        indicator_cache = _stage_build_indicator_cache(cfg, train_df, prog)

        signal_matrix, signal_names, signal_metadata, date_index = \
            _stage_generate_signal_matrix(cfg, train_df, indicator_cache, prog)

        target_returns_dict, bil_returns = _stage_compute_returns(cfg, train_df, prog)

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
            train_df, bil_returns, prog
        )

        all_signals_df = _stage_aggregate_oos(oos_raw_df, prog)

        all_signals_df = _stage_tail_metrics(
            all_signals_df, signal_matrix, signal_names, target_returns_dict, prog
        )

        top_n_df, top_n_specs = _stage_select_top_n(
            all_signals_df, cfg.get("top_n", 50), cfg, prog
        )

        dl_cfg = cfg.get("dual_layer", {})
        if dl_cfg.get("enabled"):
            from src.orchestrator import run_dual_layer
            symphony = run_dual_layer(
                cfg=cfg,
                train_df=train_df,
                holdout_df=holdout_df,
                signal_matrix=signal_matrix,
                signal_names=signal_names,
                signal_metadata=signal_metadata,
                top_n_df=top_n_df,
                top_n_specs=top_n_specs,
                target_returns_dict=target_returns_dict,
                bil_returns=bil_returns,
                output_dir=run_dir,
            )
        else:
            symphony = _stage_build_symphony(top_n_specs, prog)

        if insert_into:
            symphony = _stage_mode_c(
                symphony, insert_into,
                insert_mode=cfg.get("insert_mode", "leaf"),
                top_n_specs=top_n_specs,
                safe_asset=cfg.get("safe_asset_ticker", "BIL"),
                prog=prog,
            )

        verify_results = _stage_verify_symphony(
            symphony, signal_matrix, signal_names, train_df, prog
        )

        if run_mc:
            _stage_monte_carlo(
                cfg, top_n_df, signal_matrix, signal_names,
                target_returns_dict, bil_returns, date_index,
                run_dir, prog
            )

        # Use date_index (train-aligned) not dates (full price history) —
        # signal_matrix and returns are always computed on train_df only.
        output_dates = date_index
        paths = _stage_write_output(
            cfg, all_signals_df, top_n_df, symphony,
            signal_matrix, signal_names, target_returns_dict, bil_returns, output_dates,
            base_output_dir,
            combo_df if run_combos else None,
            prog,
            run_timestamp=run_timestamp,
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
