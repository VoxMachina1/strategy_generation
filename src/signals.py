"""
Signal generation layer for the Composer Signal Pipeline.

Public API
----------
SignalSpec               — dataclass describing one signal/target pair
make_signal_name()       — build the canonical string name for a signal
parse_signal_name()      — reconstruct config fields from a signal name string
generate_signal_specs()  — expand a config dict into a list of SignalSpec
derive_required_indicators() — extract unique (ticker, fn, window) tuples
generate_signal_matrix() — build the boolean (n_days × n_signals) numpy matrix

Naming contract (locked — downstream Composer export depends on this)
----------------------------------------------------------------------
Fixed threshold:     {fn}_{window}_{ticker}_{COMPARATOR}_{threshold}
                     e.g. RSI_10_SPY_GT_50
Indicator vs indicator: {fn}_{window}_{ticker}_{COMPARATOR}_{rhs_fn}_{rhs_window}_{rhs_ticker}
                     e.g. SMA_20_QQQ_GT_SMA_20_TLT

Comparators (4 only — crosses_above/crosses_below are out of scope):
  "gt"  → GT   "lt"  → LT   "gte" → GTE   "lte" → LTE
"""

import itertools

import numpy as np
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

KNOWN_COMPS: frozenset = frozenset({"GT", "LT", "GTE", "LTE"})

_COMP_MAP: dict = {
    "gt": "GT",
    "lt": "LT",
    "gte": "GTE",
    "lte": "LTE",
}


# ---------------------------------------------------------------------------
# SignalSpec dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalSpec:
    """
    Describes one signal/target pair — the atomic unit of the signal matrix.

    Fields
    ------
    name        : canonical string built by make_signal_name()
    lhs_ticker  : ticker whose indicator forms the left-hand side
    lhs_fn      : indicator label: "RSI", "SMA", "EMA", "CumRet", "MaxDD", "MAReturn"
    lhs_window  : lookback period for the LHS indicator
    comparator  : one of "gt", "lt", "gte", "lte" (lowercase)
    rhs_type    : "fixed" (scalar threshold) or "indicator" (compare to another indicator)
    rhs_value   : scalar threshold — set when rhs_type == "fixed", else None
    rhs_ticker  : RHS ticker — set when rhs_type == "indicator", else None
    rhs_fn      : RHS indicator label — set when rhs_type == "indicator", else None
    rhs_window  : RHS lookback period — set when rhs_type == "indicator", else None
    target      : allocation ticker held when this signal fires (e.g. "TQQQ")
    """

    name: str
    lhs_ticker: str
    lhs_fn: str
    lhs_window: int
    comparator: str
    rhs_type: str
    rhs_value: Optional[float]
    rhs_ticker: Optional[str]
    rhs_fn: Optional[str]
    rhs_window: Optional[int]
    target: str


# ---------------------------------------------------------------------------
# Naming
# ---------------------------------------------------------------------------

def make_signal_name(
    lhs_fn: str,
    lhs_window: int,
    lhs_ticker: str,
    comparator: str,
    rhs_type: str,
    rhs_value: Optional[float] = None,
    rhs_fn: Optional[str] = None,
    rhs_window: Optional[int] = None,
    rhs_ticker: Optional[str] = None,
) -> str:
    """
    Build the canonical signal name string.

    Fixed threshold:
        make_signal_name("RSI", 10, "SPY", "gt", "fixed", rhs_value=50.0)
        → "RSI_10_SPY_GT_50"

    Indicator vs indicator:
        make_signal_name("SMA", 20, "QQQ", "gt", "indicator",
                         rhs_fn="SMA", rhs_window=20, rhs_ticker="TLT")
        → "SMA_20_QQQ_GT_SMA_20_TLT"

    Integer thresholds drop the trailing .0 ("50" not "50.0").
    Genuine decimal thresholds are preserved ("0.02" not "0").
    """
    comp_token = _COMP_MAP[comparator]
    lhs = f"{lhs_fn}_{lhs_window}_{lhs_ticker}"
    if rhs_type == "fixed":
        rhs_str = str(int(rhs_value)) if rhs_value == int(rhs_value) else str(rhs_value)
        return f"{lhs}_{comp_token}_{rhs_str}"
    else:
        return f"{lhs}_{comp_token}_{rhs_fn}_{rhs_window}_{rhs_ticker}"


def parse_signal_name(name: str) -> dict:
    """
    Reconstruct config fields from a canonical signal name string.

    Returns a dict with keys:
        lhs_fn, lhs_window, lhs_ticker, comparator,
        rhs_type, rhs_value, rhs_fn, rhs_window, rhs_ticker

    Raises ValueError if the name is malformed, has an unknown comparator token,
    or has too few tokens for the detected rhs_type.

    Note: tickers must contain no underscores (e.g. "SPY", not "BRK_B").
    Combo names (containing "+") must be split before calling this function.
    """
    parts = name.split("_")
    if len(parts) < 5:
        raise ValueError(
            f"Signal name has fewer than 5 tokens: {name!r}. "
            "Expected format: {{fn}}_{{window}}_{{ticker}}_{{COMP}}_{{threshold_or_rhs}}"
        )
    lhs_fn = parts[0]
    lhs_window = int(parts[1])
    lhs_ticker = parts[2]
    comp_token = parts[3]
    if comp_token not in KNOWN_COMPS:
        raise ValueError(
            f"Unknown comparator token {comp_token!r} in signal name {name!r}. "
            f"Known tokens: {sorted(KNOWN_COMPS)}"
        )
    comparator = comp_token.lower()
    try:
        rhs_value = float(parts[4])
        return {
            "lhs_fn": lhs_fn,
            "lhs_window": lhs_window,
            "lhs_ticker": lhs_ticker,
            "comparator": comparator,
            "rhs_type": "fixed",
            "rhs_value": rhs_value,
            "rhs_fn": None,
            "rhs_window": None,
            "rhs_ticker": None,
        }
    except ValueError:
        if len(parts) < 7:
            raise ValueError(
                f"Indicator RHS requires 7 tokens, got {len(parts)} in: {name!r}"
            )
        return {
            "lhs_fn": lhs_fn,
            "lhs_window": lhs_window,
            "lhs_ticker": lhs_ticker,
            "comparator": comparator,
            "rhs_type": "indicator",
            "rhs_value": None,
            "rhs_fn": parts[4],
            "rhs_window": int(parts[5]),
            "rhs_ticker": parts[6],
        }


# ---------------------------------------------------------------------------
# Spec generation from config
# ---------------------------------------------------------------------------

def generate_signal_specs(config: dict) -> list:
    """
    Expand a config dict into a flat list of SignalSpec instances.

    One SignalSpec is created per (signal, target_ticker) pair.

    Config keys (all optional — missing keys produce no specs for that type):
        signal_tickers  : list[str]  — LHS tickers
        target_tickers  : list[str]  — allocation tickers (cross-producted with signals)
        rsi_windows     : list[int]
        rsi_thresholds  : list[float]
        rsi_comparators : list[str]  e.g. ["lt", "gt"]
        sma_windows     : list[int]
        ema_windows     : list[int]
        cross_tickers   : list[str]  — RHS tickers for indicator-vs-indicator signals

    Unit note — CumRet thresholds must be percentage-scale to match
    calculate_cumret() output (e.g. 5.0 means 5%, not 0.05). Using
    decimal-scale thresholds (0.05) produces silently wrong signals.
    """
    signal_tickers = config.get("signal_tickers", [])
    target_tickers = config.get("target_tickers", [])
    rsi_windows = config.get("rsi_windows", [])
    rsi_thresholds = config.get("rsi_thresholds", [])
    rsi_comparators = config.get("rsi_comparators", [])
    sma_windows = config.get("sma_windows", [])
    ema_windows = config.get("ema_windows", [])
    cross_tickers = config.get("cross_tickers", [])

    specs = []

    # RSI fixed-threshold signals
    for ticker, window, threshold, comp, target in itertools.product(
        signal_tickers, rsi_windows, rsi_thresholds, rsi_comparators, target_tickers
    ):
        name = make_signal_name("RSI", window, ticker, comp, "fixed", rhs_value=float(threshold))
        specs.append(SignalSpec(
            name=name,
            lhs_ticker=ticker,
            lhs_fn="RSI",
            lhs_window=window,
            comparator=comp,
            rhs_type="fixed",
            rhs_value=float(threshold),
            rhs_ticker=None,
            rhs_fn=None,
            rhs_window=None,
            target=target,
        ))

    # SMA cross signals (indicator vs indicator, gt only per PLANNING.md §2.3)
    for lhs_ticker, lhs_w, rhs_ticker, rhs_w, target in itertools.product(
        signal_tickers, sma_windows, cross_tickers, sma_windows, target_tickers
    ):
        if lhs_ticker == rhs_ticker and lhs_w == rhs_w:
            continue  # self-comparison: always True, useless signal
        name = make_signal_name(
            "SMA", lhs_w, lhs_ticker, "gt", "indicator",
            rhs_fn="SMA", rhs_window=rhs_w, rhs_ticker=rhs_ticker,
        )
        specs.append(SignalSpec(
            name=name,
            lhs_ticker=lhs_ticker,
            lhs_fn="SMA",
            lhs_window=lhs_w,
            comparator="gt",
            rhs_type="indicator",
            rhs_value=None,
            rhs_ticker=rhs_ticker,
            rhs_fn="SMA",
            rhs_window=rhs_w,
            target=target,
        ))

    # EMA cross signals (indicator vs indicator, gt only)
    for lhs_ticker, lhs_w, rhs_ticker, rhs_w, target in itertools.product(
        signal_tickers, ema_windows, cross_tickers, ema_windows, target_tickers
    ):
        if lhs_ticker == rhs_ticker and lhs_w == rhs_w:
            continue  # self-comparison: always True, useless signal
        name = make_signal_name(
            "EMA", lhs_w, lhs_ticker, "gt", "indicator",
            rhs_fn="EMA", rhs_window=rhs_w, rhs_ticker=rhs_ticker,
        )
        specs.append(SignalSpec(
            name=name,
            lhs_ticker=lhs_ticker,
            lhs_fn="EMA",
            lhs_window=lhs_w,
            comparator="gt",
            rhs_type="indicator",
            rhs_value=None,
            rhs_ticker=rhs_ticker,
            rhs_fn="EMA",
            rhs_window=rhs_w,
            target=target,
        ))

    # -------------------------------------------------------------------------
    # EXPERIMENTAL SIGNALS — gated by experimental_signals=True in config.
    # These indicators are not yet supported by Composer and must not appear
    # in production symphony exports. Enable only for research/testing.
    # -------------------------------------------------------------------------
    if config.get("experimental_signals", False):
        macd_params_list = config.get("macd_params", [(12, 26, 9)])
        bband_windows = config.get("bband_windows", [])
        bband_std = config.get("bband_std", 2.0)

        # MACD histogram signals: histogram gt 0 (bullish cross) / lt 0 (bearish cross)
        # Cache key uses the param tuple as window: (ticker, "MACD", (fast, slow, signal))
        for ticker, params, comp, target in itertools.product(
            signal_tickers, macd_params_list, ["gt", "lt"], target_tickers
        ):
            fast, slow, sig_p = params
            param_str = f"{fast}_{slow}_{sig_p}"
            name = f"MACD_{param_str}_{ticker}_{comp.upper()}_0"
            specs.append(SignalSpec(
                name=name,
                lhs_ticker=ticker,
                lhs_fn="MACD",
                lhs_window=tuple(params),  # (fast, slow, signal_period)
                comparator=comp,
                rhs_type="fixed",
                rhs_value=0.0,
                rhs_ticker=None,
                rhs_fn=None,
                rhs_window=None,
                target=target,
            ))

        # Bollinger Band signals: price lt lower band / price gt upper band
        # Modelled as indicator-vs-indicator where LHS is the price series (SMA_1 proxy)
        # and RHS is the band. Cache uses window=bband_window for both.
        for ticker, window, comp, target in itertools.product(
            signal_tickers, bband_windows, ["lt", "gt"], target_tickers
        ):
            band_fn = "BBAND_LOWER" if comp == "lt" else "BBAND_UPPER"
            name = f"BBAND_{window}_{ticker}_{comp.upper()}_{band_fn}"
            specs.append(SignalSpec(
                name=name,
                lhs_ticker=ticker,
                lhs_fn="SMA",
                lhs_window=1,          # SMA(1) == close price, deduplicated in cache
                comparator=comp,
                rhs_type="indicator",
                rhs_value=None,
                rhs_ticker=ticker,
                rhs_fn=band_fn,
                rhs_window=window,
                target=target,
            ))

    return specs


# ---------------------------------------------------------------------------
# Indicator requirement derivation
# ---------------------------------------------------------------------------

def derive_required_indicators(specs: list) -> list:
    """
    Extract all unique (ticker, fn, window) tuples needed by a spec list.

    Covers both LHS and RHS sides. Deduplicates: each unique triple appears once.
    The returned list has no guaranteed order.

    Feeds directly into build_indicator_cache(price_df, derive_required_indicators(specs)).
    """
    seen = set()
    for spec in specs:
        seen.add((spec.lhs_ticker, spec.lhs_fn, spec.lhs_window))
        if spec.rhs_type == "indicator":
            seen.add((spec.rhs_ticker, spec.rhs_fn, spec.rhs_window))
    return list(seen)


# ---------------------------------------------------------------------------
# Boolean matrix construction
# ---------------------------------------------------------------------------

def _evaluate_spec(spec: SignalSpec, indicator_cache: dict) -> np.ndarray:
    """
    Evaluate one SignalSpec against the indicator cache.

    Returns a 1D boolean array of shape (n_days,).
    NaN positions in either LHS or RHS array produce False (never True or NaN).

    Raises KeyError if a required cache entry is missing.
    """
    lhs_arr = indicator_cache[(spec.lhs_ticker, spec.lhs_fn, spec.lhs_window)]

    if spec.rhs_type == "fixed":
        rhs_arr = spec.rhs_value  # scalar broadcast
    else:
        rhs_arr = indicator_cache[(spec.rhs_ticker, spec.rhs_fn, spec.rhs_window)]

    if spec.comparator == "gt":
        col = lhs_arr > rhs_arr
    elif spec.comparator == "lt":
        col = lhs_arr < rhs_arr
    elif spec.comparator == "gte":
        col = lhs_arr >= rhs_arr
    elif spec.comparator == "lte":
        col = lhs_arr <= rhs_arr
    else:
        raise ValueError(f"Unknown comparator: {spec.comparator!r}")

    # Explicit NaN → False guarantee.
    # (numpy comparisons with NaN already return False, but this makes the
    # contract visible and guards against scalar-rhs edge cases.)
    nan_mask = np.isnan(lhs_arr)
    if isinstance(rhs_arr, np.ndarray):
        nan_mask = nan_mask | np.isnan(rhs_arr)
    col = np.where(nan_mask, False, col)
    return col.astype(bool)


def generate_signal_matrix(
    specs: list,
    indicator_cache: dict,
    date_index: np.ndarray,
) -> tuple:
    """
    Build the boolean signal matrix from a list of SignalSpec instances.

    Parameters
    ----------
    specs           : list[SignalSpec] — one per signal/target pair
    indicator_cache : dict[(ticker, fn, window) → np.ndarray[float64, n_days]]
                      Built by build_indicator_cache() in src/data/cache.py.
    date_index      : np.ndarray of shape (n_days,) — calendar dates.
                      Passed in rather than derived here; use price_df.index.to_numpy().

    Returns
    -------
    signal_matrix   : np.ndarray[bool], shape (n_days, n_signals)
    signal_names    : list[str], parallel to columns
    signal_metadata : list[SignalSpec], parallel to columns

    Contract: NaN positions in any indicator array produce False in the
    corresponding matrix column — never True or NaN.

    Raises KeyError if indicator_cache is missing a required entry.
    Callers should build the cache via:
        build_indicator_cache(price_df, derive_required_indicators(specs))
    """
    if not specs:
        return np.empty((len(date_index), 0), dtype=bool), [], []

    bool_cols = [_evaluate_spec(spec, indicator_cache) for spec in specs]
    signal_matrix = np.column_stack(bool_cols).astype(bool)
    return signal_matrix, [spec.name for spec in specs], list(specs)
