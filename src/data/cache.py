import numpy as np
import pandas as pd
from src.indicators import (
    calculate_rsi,
    calculate_sma,
    calculate_ema,
    calculate_cumret,
    calculate_atr,
    calculate_bbands_lower,
    calculate_maxdd,
    calculate_mareturn,
)


def _compute_indicator(series, fn, window, price_df, ticker):
    """
    Dispatches a single-series indicator computation by label.

    Args:
        series:    pd.Series of close prices for the ticker
        fn:        indicator label string (case-insensitive)
        window:    integer lookback period
        price_df:  the full price DataFrame (needed for future ATR dispatch)
        ticker:    ticker symbol string

    Returns:
        pd.Series of computed indicator values

    Raises:
        NotImplementedError: for ATR (Phase 2 will add OHLC dispatch)
        ValueError: for unrecognized indicator labels
    """
    fn_upper = fn.upper()
    if fn_upper == "RSI":
        return calculate_rsi(series, window)
    elif fn_upper == "SMA":
        return calculate_sma(series, window)
    elif fn_upper == "EMA":
        return calculate_ema(series, window)
    elif fn_upper == "CUMRET":
        return calculate_cumret(series, window)
    elif fn_upper == "ATR":
        raise NotImplementedError(
            "ATR dispatch in build_indicator_cache requires OHLC columns — "
            "deferred to Phase 2. Use calculate_atr() directly with high/low/close Series."
        )
    elif fn_upper == "BBAND_LOWER":
        return calculate_bbands_lower(series, window)
    elif fn_upper == "MAXDD":
        return calculate_maxdd(series, window)
    elif fn_upper == "MARETURN":
        return calculate_mareturn(series, window)
    else:
        raise ValueError(f"Unknown indicator: {fn!r}")


def build_indicator_cache(price_df, required):
    """
    Pre-computes indicator series for all (ticker, fn_label, window) requests.
    Deduplicates: each unique (ticker, fn_label, window) key is computed exactly once.

    Args:
        price_df:  DataFrame with columns = ticker symbols, index = date (DatetimeIndex).
                   This is the output of load_multi_ticker_aligned().
        required:  List of (ticker, fn_label, window) tuples. May contain duplicates.

    Returns:
        dict mapping (ticker, fn_label, window) -> np.ndarray of shape (n_days,).
        Values are float64. NaN values are preserved (warmup periods, etc.).

    Raises:
        KeyError: if a ticker in required is not a column in price_df
        NotImplementedError: if fn_label is "ATR" (Phase 2)
        ValueError: if fn_label is unrecognized
    """
    cache = {}
    seen = set()
    for ticker, fn, window in required:
        key = (ticker, fn, window)
        if key in seen:
            continue
        seen.add(key)
        series = price_df[ticker]
        indicator_series = _compute_indicator(series, fn, window, price_df, ticker)
        cache[key] = indicator_series.to_numpy(dtype=float)
    return cache
