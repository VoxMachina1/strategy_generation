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

from pathlib import Path

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
    ],
    # Tickers to hold while the signal is active. BIL is the safe-asset
    # fallback — keep it in this list so it appears as an allocation option.
    "target_tickers": ["TQQQ", "SQQQ", "TLT", "BIL"],
    # What the strategy holds when NO signal is firing (the "else" branch).
    # This is the cash/safe-asset position — typically BIL or a short-duration
    # bond fund. It is NOT the same as benchmark_ticker (see below).
    "safe_asset_ticker": "BIL",
    # What strategy performance is MEASURED AGAINST for Sharpe and other
    # relative metrics. Does not affect what the strategy holds — only affects
    # how results are evaluated. Set to SPY to ask "did this beat the market?"
    "benchmark_ticker": "SPY",
    # -----------------------------------------------------------------------
    # RSI signals
    # The pipeline generates one signal per combination of
    # (signal_ticker × rsi_window × rsi_threshold × comparator × target_ticker).
    # -----------------------------------------------------------------------
    "rsi_windows": [5, 10, 15, 20],
    # RSI threshold levels to sweep. Must match the scale output by
    # calculate_rsi() — i.e. 0–100, not 0–1.
    "rsi_thresholds": [20, 30, 40, 50, 60, 70, 80],
    # "lt" → RSI < threshold (oversold / re-entry signals)
    # "gt" → RSI > threshold (momentum / overbought signals)
    "comparators": ["lt", "gt"],
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
    # Whether to run Monte Carlo walk-forward simulation for top-N signals.
    # Disabling cuts runtime by several minutes on large top-N values.
    "run_mc": True,
}

# ---------------------------------------------------------------------------
# Backwards-compatibility alias — rsi_search.py imports RSI_SEARCH_CONFIG
# ---------------------------------------------------------------------------
RSI_SEARCH_CONFIG = PIPELINE_CONFIG
