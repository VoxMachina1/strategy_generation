"""
Central configuration for the Composer Signal Pipeline.

Edit this file before running any entry point. All three entry points read
from PIPELINE_CONFIG:

    python main.py              — full discovery pipeline
    python rsi_search.py        — fast RSI parameter sweep
    python analysis_workshop.py — interactive filter/sort CLI

CLI flags override the values here for a single run. To make a change
permanent, edit this file.
"""

import os
from pathlib import Path


def _auto_workers() -> int:
    """
    Choose a safe default worker count using available CPU cores and free RAM.
    Each parallel sub-run uses roughly 250 MB; we cap below that.
    Falls back to a CPU-only heuristic if psutil is not installed.
    """
    cores = os.cpu_count() or 4
    try:
        import psutil

        available_gb = psutil.virtual_memory().available / (1024**3)
        mem_cap = max(1, int(available_gb / 0.25))
    except ImportError:
        mem_cap = 4
    return max(1, min(cores - 2, mem_cap, 12))


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "prices"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Main config — edit this before running
# ---------------------------------------------------------------------------

PIPELINE_CONFIG = {
    # -----------------------------------------------------------------------
    # Universe
    # Tickers whose indicators are evaluated as signal triggers.
    # -----------------------------------------------------------------------
    "signal_tickers": [
        "SPY",
        "SPYV",
        "IOO",
        "VTV",
        "QQQ",
        "QQQE",
        "XLF",
        "XLK",
        "XLE",
        "XLY",
        "XLP",
        "TLT",
        "USO",
        "CORP",
        "GLD",
        "IEF",
        "IGIB",
        "BDRY",
        "HYG",
        "AGG",
        "BND",
        "DLN",
        "QQEW",
        "RSPS",
        "IDGT",
        "DEW",
        "TIP",
        "FXU",
        "IYK",
        "VIXY",
    ],
    # Tickers to hold WHEN a signal fires (the true branch of each if-block).
    # The pipeline runs a separate signal search for each ticker independently,
    # so including multiple targets generates signals for each in parallel.
    # Do NOT include safe_asset_ticker here — the else branch already holds it,
    # and including it inflates Sharpe (near-zero volatility in the denominator).
    #
    # Example — leveraged Nasdaq trades in both directions:
    #   ["TQQQ", "SQQQ"] finds signals predicting up-moves in each independently.
    #   This is NOT a risk-on/risk-off setup; it is two separate signal searches.
    #
    # Example — risk-on / risk-off for the Nasdaq:
    #   ["QQQ"] with safe_asset_ticker="BIL" and benchmark_ticker="SPY"
    #   asks "when should I be in QQQ vs cash, and does that beat the market?"
    #
    # Example — high-probability volatility trade:
    #   ["UVXY"] finds signals that predict volatility spikes (UVXY up-moves).
    "target_tickers": ["UVXY"],
    # What the strategy holds when NO signal is firing (the "else" branch).
    # This is the cash/safe-asset position — typically BIL or a short-duration
    # bond fund. It is NOT the same as benchmark_ticker (see below).
    "safe_asset_ticker": "QQQ",
    # What strategy performance is MEASURED AGAINST for Sharpe and other
    # relative metrics. Does not affect what the strategy holds — only affects
    # how results are evaluated. Set to SPY to ask "did this beat the market?"
    #
    # It is valid — and often correct — for safe_asset_ticker and
    # benchmark_ticker to be the same ticker. If the null hypothesis is
    # "I would otherwise be holding QQQ," set both to "QQQ" so the pipeline
    # measures whether signals beat that baseline.
    "benchmark_ticker": "qqq",
    # -----------------------------------------------------------------------
    # RSI signals
    # The pipeline generates one signal per combination of
    # (signal_ticker × rsi_window × rsi_threshold × comparator × target_ticker).
    # -----------------------------------------------------------------------
    "rsi_windows": [5, 10, 15, 20],
    # RSI threshold levels to sweep. Must match the scale output by
    # calculate_rsi() — i.e. 0–100, not 0–1.
    "rsi_thresholds": [20, 25, 30, 40, 50, 60, 70, 80, 82, 84, 86, 88, 90],
    # "lt" → RSI < threshold (oversold / re-entry signals)
    # "gt" → RSI > threshold (momentum / overbought signals)
    "comparators": ["lt", "gt"],
    # -----------------------------------------------------------------------
    # Moving average crossover signals
    # Each combination of (lhs_ticker, lhs_window, rhs_ticker, rhs_window)
    # produces one signal: lhs_MA > rhs_MA. SMA(1) = current price, so
    # SMA_1_SPY_GT_SMA_200_SPY means "SPY price above its 200-day MA."
    # Both sides draw from signal_tickers — no separate cross_tickers needed.
    # -----------------------------------------------------------------------
    "sma_windows": [1, 20, 50, 200],
    "ema_windows": [1, 12, 26, 50],
    # -----------------------------------------------------------------------
    # Experimental signals (main.py only)
    # MACD and Bollinger Band signals are not yet supported by Composer.
    # Set to True only for local research — never export experimental signals
    # to a symphony intended for live trading.
    # -----------------------------------------------------------------------
    "experimental_signals": False,
    # MACD histogram (fast EMA - slow EMA) minus signal EMA.
    # Each tuple is (fast_period, slow_period, signal_period).
    # Only evaluated when experimental_signals=True.
    "macd_params": [(12, 26, 9)],
    # Bollinger Band windows to sweep (price vs upper/lower band).
    # Only evaluated when experimental_signals=True.
    "bband_windows": [20],
    # Standard deviations for Bollinger Band width.
    "bband_std": 2.0,
    # -----------------------------------------------------------------------
    # RSI Search filters (rsi_search.py only)
    # These apply when running the standalone RSI sweep; they are not used
    # by the full validation pipeline in main.py.
    # -----------------------------------------------------------------------
    # Minimum number of active signal days to include a row in rsi_search output.
    "min_trades": 20,
    # Minimum win-rate to include a row. Win-rate = fraction of signal days
    # where target return > BIL return.
    "min_win_rate": 0.55,
    # If True, rows where the benchmark had a negative median return during
    # signal days are excluded. If False (default), they are flagged in
    # the Benchmark_Negative column but still appear in output.
    "filter_benchmark_negative": False,
    # -----------------------------------------------------------------------
    # Validation (main.py)
    # Controls how out-of-sample windows are constructed.
    # -----------------------------------------------------------------------
    "validation": {
        # "walk_forward" — fixed-length train, non-overlapping test windows (default)
        # "expanding"    — growing train window, fixed-length test windows
        # "rolling"      — fixed-length train, fixed-length test, sliding step
        "window_type": "walk_forward",
        # Number of trading days in each training window (~3 years = 756)
        "train_size": 756,
        # Number of trading days in each test window (~1 quarter = 63)
        "test_size": 63,
    },
    # -----------------------------------------------------------------------
    # Pipeline behaviour (main.py)
    # -----------------------------------------------------------------------
    # Number of top signals to include in the Composer symphony output.
    "top_n": 50,
    # -----------------------------------------------------------------------
    # Top-N quality filters
    # Applied before ranking — signals that fail any threshold are excluded
    # from the report entirely, regardless of Sharpe.
    # -----------------------------------------------------------------------
    # Win rate after removing tail events. Filters out lottery-ticket signals
    # that only look good because of a handful of rare lucky days.
    "min_stripped_win_rate": 0.55,
    # Raw win rate on all active days (including tail events).
    "min_base_win_rate": 0.45,
    # Maximum fraction of total return attributable to tail events.
    # Set permissively to avoid filtering legitimate momentum signals.
    "max_tail_concentration": 0.80,
    # Fraction of OOS windows where Sharpe > 0. Filters regime-dependent signals.
    "min_consistency_score": 0.60,
    # Minimum number of OOS windows required. Below this there is no statistical
    # basis for ranking the signal.
    "min_n_iterations": 5,
    # Whether to generate and backtest pairwise signal combinations (AND/OR/etc.).
    # Disabling cuts runtime significantly on large universes.
    "run_combos": True,
    # Cap on the number of signals considered for combo pairing.
    # Keeps C(K,2)×4 tractable. This is NOT a quality filter — quality
    # gates are applied to combo results after the fact.
    "top_k_for_combos": 50,
    # Number of combo columns processed per batch. Lower values use less
    # peak memory; higher values are faster. 500 is a safe default.
    "combo_batch_size": 500,
    # Write a standalone Composer symphony JSON for every (signal, target)
    # and (combo, target) row -- the FULL unfiltered universe, not just
    # top-N. Top-N depends on quality thresholds tuned for the report;
    # for ad-hoc inspection of the raw dataset (e.g. hunting a specific
    # regime signal by hand) that filtering is actively unhelpful.
    # Expensive: with the default universe this is tens of thousands of
    # small files and adds real wall-clock time to every run. Default off.
    # Files land in output/{timestamp}/composer_json/{name}__{target}.json.
    "export_individual_json": False,
    # Whether to run Monte Carlo walk-forward simulation for top-N signals.
    # Disabling cuts runtime by several minutes on large top-N values.
    "run_mc": True,
    # Global parallel worker count for all pipeline stages (MC, validation, dual-layer).
    # Auto-detected from CPU cores and available RAM; override here or via --workers.
    "workers": _auto_workers(),
    # -----------------------------------------------------------------------
    # Dual-layer 3-pass architecture (Pass 2 + Pass 3)
    # Set enabled=True to run the offense/defense/else pipeline after Pass 1.
    # See VISION.md "Dual-layer 3-pass architecture" for design rationale.
    # -----------------------------------------------------------------------
    "dual_layer": {
        # Master switch — False means only Pass 1 (standard pipeline) runs.
        "enabled": True,
        # Assets the pipeline searches for defense signals (Pass 2) and
        # else-state assets (Pass 3). May overlap with target_tickers.
        "defensive_target_tickers": ["GLD", "TLT", "SH", "BIL"],
        # ISO-8601 date string. Price data on or after this date is held out
        # from all three passes and used only for final validation.
        # Per VISION.md: any ETF-backfill "extended" period is also treated
        # as holdout regardless of this cutoff.
        "holdout_cutoff": "2023-01-01",
        # How many precondition blocks to include in the Composer JSON output.
        # The full top_n results are always written to the CSV/HTML report.
        "composer_top_n": 15,
        # Coefficient weighting max-drawdown against Sharpe in defense scoring.
        # Higher values penalise defense signals that have large drawdowns more.
        "defense_drawdown_penalty": 0.5,
    },
}

# ---------------------------------------------------------------------------
# Backwards-compatibility alias — rsi_search.py imports RSI_SEARCH_CONFIG
# ---------------------------------------------------------------------------
RSI_SEARCH_CONFIG = PIPELINE_CONFIG
