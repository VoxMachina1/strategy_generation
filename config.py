"""
Central configuration for the Composer Signal Pipeline.

Edit this file before running any entry point. All paths are relative to the
project root. Data files are downloaded to data_dir on first run.

RSI Search path (rsi_search.py) only uses: RSI_SEARCH_CONFIG.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
DATA_DIR     = PROJECT_ROOT / "data" / "prices"
OUTPUT_DIR   = PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# RSI Search config
# ---------------------------------------------------------------------------

RSI_SEARCH_CONFIG = {
    # Tickers whose RSI is evaluated as the signal trigger
    "signal_tickers": ["SPY", "QQQ", "IWM"],

    # Tickers to hold while the signal is active
    "target_tickers": ["TQQQ", "SQQQ", "TLT", "BIL"],

    # RSI windows to sweep
    "rsi_windows": [5, 10, 14],

    # RSI threshold levels to sweep
    "rsi_thresholds": [20, 30, 40, 50, 60, 70, 80],

    # Comparators: "lt" (RSI < threshold) or "gt" (RSI > threshold)
    "comparators": ["lt", "gt"],

    # Benchmark ticker — typically BIL or cash proxy
    "benchmark_ticker": "BIL",

    # Minimum number of signal days required to include a row in output
    "min_trades": 20,

    # Minimum win-rate required to include a row in output
    # Win-rate = fraction of signal days where target > BIL
    "min_win_rate": 0.55,

    # If True, rows where Benchmark_Median_Return < 0 are excluded from output.
    # If False (default), they appear in the output with Benchmark_Negative=True.
    "filter_benchmark_negative": False,
}
