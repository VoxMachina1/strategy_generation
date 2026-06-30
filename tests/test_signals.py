"""
Tests for src/signals.py — Phase 2: Signal Matrix Generation.

Covers SC-1 through SC-4:
  SC-1: generate_signal_matrix() returns correct shape and dtype
  SC-2: Signal names follow the {fn}_{window}_{ticker}_{COMP}_{rhs} contract
  SC-3: NaN warmup days produce False in the matrix (not True or NaN)
  SC-4: parse_signal_name(make_signal_name(...)) round-trips for all 4 comparators
"""

import numpy as np
import pytest

from src.signals import (
    SignalSpec,
    derive_required_indicators,
    generate_signal_matrix,
    generate_signal_specs,
    make_signal_name,
    parse_signal_name,
)


# ---------------------------------------------------------------------------
# SC-2: make_signal_name
# ---------------------------------------------------------------------------

def test_make_signal_name_fixed():
    """Integer thresholds drop trailing .0; decimal thresholds are preserved."""
    # Integer case: 50.0 → "50" not "50.0"
    assert make_signal_name("RSI", 10, "SPY", "gt", "fixed", rhs_value=50.0) == "RSI_10_SPY_GT_50"
    # Decimal case: 0.02 → "0.02" not "0"
    assert make_signal_name("CumRet", 20, "SPY", "lt", "fixed", rhs_value=0.02) == "CumRet_20_SPY_LT_0.02"
    # LTE / GTE tokens
    assert make_signal_name("RSI", 14, "QQQ", "gte", "fixed", rhs_value=30.0) == "RSI_14_QQQ_GTE_30"
    assert make_signal_name("RSI", 14, "QQQ", "lte", "fixed", rhs_value=70.0) == "RSI_14_QQQ_LTE_70"


def test_make_signal_name_indicator():
    """Indicator-vs-indicator format: {fn}_{w}_{t}_{COMP}_{rhs_fn}_{rhs_w}_{rhs_t}."""
    assert (
        make_signal_name("SMA", 20, "QQQ", "gt", "indicator",
                         rhs_fn="SMA", rhs_window=20, rhs_ticker="TLT")
        == "SMA_20_QQQ_GT_SMA_20_TLT"
    )
    # Different windows, same ticker (golden cross)
    assert (
        make_signal_name("EMA", 50, "IWM", "gt", "indicator",
                         rhs_fn="EMA", rhs_window=200, rhs_ticker="IWM")
        == "EMA_50_IWM_GT_EMA_200_IWM"
    )


# ---------------------------------------------------------------------------
# SC-4: parse_signal_name round-trips
# ---------------------------------------------------------------------------

def test_parse_round_trip_fixed():
    """parse_signal_name(make_signal_name(...)) recovers original fields for all 4 comparators."""
    for comp in ("gt", "lt", "gte", "lte"):
        name = make_signal_name("RSI", 14, "QQQ", comp, "fixed", rhs_value=70.0)
        parsed = parse_signal_name(name)
        assert parsed["lhs_fn"] == "RSI"
        assert parsed["lhs_window"] == 14
        assert parsed["lhs_ticker"] == "QQQ"
        assert parsed["comparator"] == comp
        assert parsed["rhs_type"] == "fixed"
        assert parsed["rhs_value"] == 70.0
        assert parsed["rhs_fn"] is None
        assert parsed["rhs_ticker"] is None
        assert parsed["rhs_window"] is None


def test_parse_round_trip_indicator():
    """parse_signal_name round-trips correctly for indicator rhs_type."""
    name = make_signal_name("SMA", 50, "SPY", "gt", "indicator",
                            rhs_fn="SMA", rhs_window=200, rhs_ticker="QQQ")
    parsed = parse_signal_name(name)
    assert parsed["lhs_fn"] == "SMA"
    assert parsed["lhs_window"] == 50
    assert parsed["lhs_ticker"] == "SPY"
    assert parsed["comparator"] == "gt"
    assert parsed["rhs_type"] == "indicator"
    assert parsed["rhs_fn"] == "SMA"
    assert parsed["rhs_window"] == 200
    assert parsed["rhs_ticker"] == "QQQ"
    assert parsed["rhs_value"] is None


def test_parse_signal_name_errors():
    """parse_signal_name raises ValueError on malformed or unknown comparator."""
    with pytest.raises(ValueError, match="fewer than 5"):
        parse_signal_name("RSI_10_SPY")
    with pytest.raises(ValueError, match="Unknown comparator"):
        parse_signal_name("RSI_10_SPY_CROSS_50")


# ---------------------------------------------------------------------------
# SC-1: generate_signal_matrix shape and dtype
# ---------------------------------------------------------------------------

def test_generate_signal_matrix_shape():
    """Matrix shape is (n_days, n_signals) with dtype bool."""
    config = {
        "signal_tickers": ["SPY", "QQQ"],
        "target_tickers": ["TQQQ"],
        "rsi_windows": [10],
        "rsi_thresholds": [50],
        "rsi_comparators": ["gt"],
        "sma_windows": [],
        "ema_windows": [],
    }
    specs = generate_signal_specs(config)
    assert len(specs) == 2  # SPY×TQQQ, QQQ×TQQQ

    n_days = 100
    cache = {
        ("SPY", "RSI", 10): np.full(n_days, 60.0),
        ("QQQ", "RSI", 10): np.full(n_days, 40.0),
    }
    matrix, names, meta = generate_signal_matrix(specs, cache, np.arange(n_days))

    assert matrix.shape == (100, 2)
    assert matrix.dtype == bool
    assert len(names) == 2
    assert len(meta) == 2
    # SPY RSI=60 > 50 → True; QQQ RSI=40 > 50 → False
    assert matrix[:, 0].all()
    assert not matrix[:, 1].any()


# ---------------------------------------------------------------------------
# SC-3: NaN warmup → False (fixed RHS)
# ---------------------------------------------------------------------------

def test_nan_warmup_is_false():
    """NaN warmup rows in a fixed-RHS signal produce False, not True or NaN."""
    n_days = 20
    lhs_arr = np.full(n_days, 60.0)
    lhs_arr[:10] = np.nan  # first 10 rows are warmup NaN

    cache = {("SPY", "RSI", 10): lhs_arr}
    spec = SignalSpec(
        name="RSI_10_SPY_GT_50",
        lhs_ticker="SPY", lhs_fn="RSI", lhs_window=10,
        comparator="gt", rhs_type="fixed", rhs_value=50.0,
        rhs_ticker=None, rhs_fn=None, rhs_window=None,
        target="TQQQ",
    )
    matrix, _, _ = generate_signal_matrix([spec], cache, np.arange(n_days))

    assert matrix.dtype == bool
    assert not matrix[:10, 0].any()   # NaN rows → False (none are True)
    assert matrix[10:, 0].all()       # non-NaN rows: 60 > 50 → True


# ---------------------------------------------------------------------------
# SC-3: NaN warmup → False (indicator RHS)
# ---------------------------------------------------------------------------

def test_nan_warmup_indicator_rhs():
    """NaN in either LHS or RHS indicator array produces False in the output column."""
    n_days = 20
    lhs_arr = np.full(n_days, 60.0)
    rhs_arr = np.full(n_days, 40.0)
    # First 5 rows: LHS is NaN; rows 5-9: RHS is NaN
    lhs_arr[:5] = np.nan
    rhs_arr[5:10] = np.nan

    cache = {
        ("SPY", "SMA", 20): lhs_arr,
        ("QQQ", "SMA", 20): rhs_arr,
    }
    spec = SignalSpec(
        name="SMA_20_SPY_GT_SMA_20_QQQ",
        lhs_ticker="SPY", lhs_fn="SMA", lhs_window=20,
        comparator="gt", rhs_type="indicator",
        rhs_value=None, rhs_ticker="QQQ", rhs_fn="SMA", rhs_window=20,
        target="TQQQ",
    )
    matrix, _, _ = generate_signal_matrix([spec], cache, np.arange(n_days))

    assert matrix.dtype == bool
    assert not matrix[:10, 0].any()   # rows 0-9 have NaN on LHS or RHS → all False
    assert matrix[10:, 0].all()       # rows 10+: 60 > 40 → True, no NaN


# ---------------------------------------------------------------------------
# Pitfall P3: empty specs guard
# ---------------------------------------------------------------------------

def test_empty_specs_guard():
    """Empty spec list returns (n_days, 0) bool matrix without raising."""
    n_days = 50
    date_index = np.arange(n_days)
    matrix, names, meta = generate_signal_matrix([], {}, date_index)
    assert matrix.shape == (50, 0)
    assert matrix.dtype == bool
    assert names == []
    assert meta == []


# ---------------------------------------------------------------------------
# derive_required_indicators deduplication
# ---------------------------------------------------------------------------

def test_derive_required_indicators():
    """Deduplicates LHS and RHS tuples; indicator-RHS tickers are included."""
    specs = [
        SignalSpec(
            name="RSI_10_SPY_GT_50",
            lhs_ticker="SPY", lhs_fn="RSI", lhs_window=10,
            comparator="gt", rhs_type="fixed", rhs_value=50.0,
            rhs_ticker=None, rhs_fn=None, rhs_window=None, target="TQQQ",
        ),
        # Duplicate of the first — should not add a second entry
        SignalSpec(
            name="RSI_10_SPY_GT_50",
            lhs_ticker="SPY", lhs_fn="RSI", lhs_window=10,
            comparator="gt", rhs_type="fixed", rhs_value=50.0,
            rhs_ticker=None, rhs_fn=None, rhs_window=None, target="BIL",
        ),
        # Indicator RHS — should add both LHS and RHS tuples
        SignalSpec(
            name="SMA_20_QQQ_GT_SMA_20_TLT",
            lhs_ticker="QQQ", lhs_fn="SMA", lhs_window=20,
            comparator="gt", rhs_type="indicator",
            rhs_value=None, rhs_ticker="TLT", rhs_fn="SMA", rhs_window=20,
            target="TQQQ",
        ),
    ]
    required = derive_required_indicators(specs)
    required_set = set(required)
    assert ("SPY", "RSI", 10) in required_set
    assert ("QQQ", "SMA", 20) in required_set
    assert ("TLT", "SMA", 20) in required_set
    assert len(required) == 3  # deduplicated: SPY×RSI×10 appears once


# ---------------------------------------------------------------------------
# Experimental signals gate
# ---------------------------------------------------------------------------

_PRICE_DATES = np.array(
    [np.datetime64("2020-01-02") + np.timedelta64(i, "D") for i in range(300)]
)
_PRICE_CLOSE = 100.0 * np.exp(np.cumsum(np.random.default_rng(0).normal(0.0003, 0.01, 300)))


def _make_price_df():
    import pandas as pd
    return pd.DataFrame({"SPY": _PRICE_CLOSE}, index=pd.DatetimeIndex(_PRICE_DATES))


def test_experimental_signals_off_by_default():
    """No MACD or BBAND specs are generated when experimental_signals is absent/False."""
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["BIL"],
        "rsi_comparators": ["lt"],
        "rsi_windows": [10],
        "rsi_thresholds": [30],
        "macd_params": [(12, 26, 9)],
        "bband_windows": [20],
    }
    specs = generate_signal_specs(cfg)
    names = [s.name for s in specs]
    assert not any("MACD" in n for n in names)
    assert not any("BBAND" in n for n in names)


def test_experimental_signals_enabled():
    """MACD and BBAND specs are generated when experimental_signals=True."""
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["BIL"],
        "rsi_comparators": [],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "experimental_signals": True,
        "macd_params": [(12, 26, 9)],
        "bband_windows": [20],
        "bband_std": 2.0,
    }
    specs = generate_signal_specs(cfg)
    names = [s.name for s in specs]
    assert any("MACD" in n for n in names), "Expected MACD specs"
    assert any("BBAND" in n for n in names), "Expected BBAND specs"


def test_macd_signal_evaluates_correctly():
    """MACD histogram spec produces a boolean column without errors."""
    import pandas as pd
    from src.data.cache import build_indicator_cache

    price_df = _make_price_df()
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["BIL"],
        "rsi_comparators": [],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "experimental_signals": True,
        "macd_params": [(12, 26, 9)],
        "bband_windows": [],
    }
    specs = generate_signal_specs(cfg)
    required = derive_required_indicators(specs)
    cache = build_indicator_cache(price_df, required)
    date_index = price_df.index.to_numpy()
    matrix, names, _ = generate_signal_matrix(specs, cache, date_index)

    assert matrix.shape == (300, len(specs))
    assert matrix.dtype == bool
    # MACD histogram gt 0 and lt 0 should together cover most days (not all False)
    assert matrix.any(), "Expected some True values in MACD signal matrix"


def test_bband_signal_evaluates_correctly():
    """Bollinger Band specs produce boolean columns without errors."""
    import pandas as pd
    from src.data.cache import build_indicator_cache

    price_df = _make_price_df()
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["BIL"],
        "rsi_comparators": [],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "experimental_signals": True,
        "macd_params": [],
        "bband_windows": [20],
        "bband_std": 2.0,
    }
    specs = generate_signal_specs(cfg)
    required = derive_required_indicators(specs)
    cache = build_indicator_cache(price_df, required)
    date_index = price_df.index.to_numpy()
    matrix, names, _ = generate_signal_matrix(specs, cache, date_index)

    assert matrix.shape == (300, len(specs))
    assert matrix.dtype == bool


# ---------------------------------------------------------------------------
# SMA / EMA crossover signals using signal_tickers for both sides
# ---------------------------------------------------------------------------

def test_sma_specs_use_signal_tickers_for_rhs():
    """SMA crossover RHS draws from signal_tickers, not a separate cross_tickers key."""
    cfg = {
        "signal_tickers": ["SPY", "QQQ"],
        "target_tickers": ["TQQQ"],
        "sma_windows": [20, 50],
        "ema_windows": [],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "rsi_comparators": [],
    }
    specs = generate_signal_specs(cfg)
    sma_specs = [s for s in specs if s.lhs_fn == "SMA"]
    rhs_tickers = {s.rhs_ticker for s in sma_specs}
    assert "SPY" in rhs_tickers
    assert "QQQ" in rhs_tickers


def test_sma_period1_specs_generated():
    """SMA(1) specs are generated when 1 is in sma_windows (current-price proxy)."""
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["TQQQ"],
        "sma_windows": [1, 200],
        "ema_windows": [],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "rsi_comparators": [],
    }
    specs = generate_signal_specs(cfg)
    names = [s.name for s in specs]
    assert any("SMA_1_SPY_GT_SMA_200_SPY" in n for n in names), names


def test_sma_self_comparison_excluded():
    """SMA(w, ticker) vs SMA(w, ticker) is always True and must be excluded."""
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["TQQQ"],
        "sma_windows": [1, 20],
        "ema_windows": [],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "rsi_comparators": [],
    }
    specs = generate_signal_specs(cfg)
    for s in specs:
        assert not (s.lhs_ticker == s.rhs_ticker and s.lhs_window == s.rhs_window), (
            f"Self-comparison spec found: {s.name}"
        )


def test_ema_period1_specs_generated():
    """EMA(1) specs are generated when 1 is in ema_windows (current-price proxy)."""
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["TQQQ"],
        "sma_windows": [],
        "ema_windows": [1, 26],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "rsi_comparators": [],
    }
    specs = generate_signal_specs(cfg)
    names = [s.name for s in specs]
    assert any("EMA_1_SPY_GT_EMA_26_SPY" in n for n in names), names


def test_sma_ema_evaluate_correctly():
    """SMA and EMA crossover specs produce valid boolean columns end-to-end."""
    import pandas as pd
    from src.data.cache import build_indicator_cache

    price_df = _make_price_df()
    cfg = {
        "signal_tickers": ["SPY"],
        "target_tickers": ["BIL"],
        "sma_windows": [1, 20],
        "ema_windows": [1, 12],
        "rsi_windows": [],
        "rsi_thresholds": [],
        "rsi_comparators": [],
    }
    specs = generate_signal_specs(cfg)
    assert len(specs) > 0
    required = derive_required_indicators(specs)
    cache = build_indicator_cache(price_df, required)
    date_index = price_df.index.to_numpy()
    matrix, names, _ = generate_signal_matrix(specs, cache, date_index)

    assert matrix.shape == (300, len(specs))
    assert matrix.dtype == bool
    assert not matrix[0].any(), "Day 0 (warmup) should be all False"
