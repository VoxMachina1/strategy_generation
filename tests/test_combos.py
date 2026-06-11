"""
Tests for src/combos.py — Phase 4: Combo Generation and Backtesting.

Covers TASKS.md success criteria:
  SC-1: parse_combo_name correctly splits name → (members, operators)
  SC-2: apply_operator produces correct boolean results for all 4 operators
  SC-3: build_combo_batch shape and dtype
  SC-4: run_combo_backtests on 315-signal matrix with top-100 pool produces
        expected result count with no OOM (C(100,2) × 4 × n_targets)
"""

import numpy as np
import pytest

from src.combos import (
    apply_operator,
    build_combo_batch,
    make_combo_name,
    parse_combo_name,
    run_combo_backtests,
)


# ---------------------------------------------------------------------------
# SC-1: parse_combo_name
# ---------------------------------------------------------------------------

def test_parse_combo_name_single_operator():
    """Basic two-member combo splits correctly."""
    members, operators = parse_combo_name(
        "RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT"
    )
    assert members == ["RSI_10_SPY_GT_50", "SMA_20_QQQ_GT_SMA_20_TLT"]
    assert operators == ["AND"]


def test_parse_combo_name_all_operators():
    """All four operator tokens are recognised."""
    for op in ("AND", "OR", "A_AND_NOT_B", "B_AND_NOT_A"):
        members, operators = parse_combo_name(f"SIG_A+{op}+SIG_B")
        assert operators == [op]
        assert members == ["SIG_A", "SIG_B"]


def test_parse_combo_name_no_operator_raises():
    """A plain signal name (no operator) raises ValueError."""
    with pytest.raises(ValueError, match="No recognised operator"):
        parse_combo_name("RSI_10_SPY_GT_50")


def test_parse_combo_name_signal_underscores_not_confused():
    """Signal names containing underscores are not split by the operator parser."""
    name = "EMA_50_IWM_GT_EMA_200_IWM+OR+SMA_20_QQQ_GT_SMA_20_TLT"
    members, operators = parse_combo_name(name)
    assert members == ["EMA_50_IWM_GT_EMA_200_IWM", "SMA_20_QQQ_GT_SMA_20_TLT"]
    assert operators == ["OR"]


def test_make_combo_name_round_trip():
    """make_combo_name + parse_combo_name round-trips correctly."""
    name = make_combo_name("RSI_10_SPY_GT_50", "AND", "SMA_20_QQQ_GT_SMA_20_TLT")
    members, operators = parse_combo_name(name)
    assert members[0] == "RSI_10_SPY_GT_50"
    assert members[1] == "SMA_20_QQQ_GT_SMA_20_TLT"
    assert operators[0] == "AND"


# ---------------------------------------------------------------------------
# SC-2: apply_operator
# ---------------------------------------------------------------------------

def test_apply_operator_and():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    result = apply_operator(a, b, "AND")
    np.testing.assert_array_equal(result, [True, False, False, False])


def test_apply_operator_or():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    result = apply_operator(a, b, "OR")
    np.testing.assert_array_equal(result, [True, True, True, False])


def test_apply_operator_a_and_not_b():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    result = apply_operator(a, b, "A_AND_NOT_B")
    np.testing.assert_array_equal(result, [False, True, False, False])


def test_apply_operator_b_and_not_a():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    result = apply_operator(a, b, "B_AND_NOT_A")
    np.testing.assert_array_equal(result, [False, False, True, False])


def test_apply_operator_unknown_raises():
    a = np.ones(5, dtype=bool)
    b = np.ones(5, dtype=bool)
    with pytest.raises(ValueError, match="Unknown operator"):
        apply_operator(a, b, "XOR")


# ---------------------------------------------------------------------------
# SC-3: build_combo_batch
# ---------------------------------------------------------------------------

def test_build_combo_batch_shape_and_dtype():
    """Output shape is (n_days, batch_size) with bool dtype."""
    rng = np.random.default_rng(42)
    sm = rng.random((200, 10)) > 0.5
    batch = [(0, 1, "AND"), (2, 3, "OR"), (4, 5, "A_AND_NOT_B"), (6, 7, "B_AND_NOT_A")]
    result = build_combo_batch(sm, batch)
    assert result.shape == (200, 4)
    assert result.dtype == bool


def test_build_combo_batch_correctness():
    """Each column matches the direct apply_operator result."""
    rng = np.random.default_rng(7)
    sm = rng.random((50, 6)) > 0.5
    batch = [(0, 1, "AND"), (2, 3, "OR")]
    result = build_combo_batch(sm, batch)
    np.testing.assert_array_equal(result[:, 0], sm[:, 0] & sm[:, 1])
    np.testing.assert_array_equal(result[:, 1], sm[:, 2] | sm[:, 3])


# ---------------------------------------------------------------------------
# SC-4: run_combo_backtests result count and no-OOM
# ---------------------------------------------------------------------------

def test_run_combo_backtests_result_count():
    """
    With n=20 signals, top_k=10, 2 targets:
    C(10,2) × 4 operators × 2 targets = 45 × 4 × 2 = 360 result rows.
    """
    rng = np.random.default_rng(99)
    n_days, n_sigs = 300, 20
    sm = rng.random((n_days, n_sigs)) > 0.5
    signal_names = [f"SIG_{i}" for i in range(n_sigs)]
    signal_metadata = [None] * n_sigs  # not inspected by run_combo_backtests

    tr_a = rng.standard_normal(n_days) * 0.01
    tr_b = rng.standard_normal(n_days) * 0.01
    target_returns_dict = {"TQQQ": tr_a, "BIL": tr_b}
    bil = np.full(n_days, 0.0001)

    results = run_combo_backtests(
        sm, signal_names, signal_metadata,
        target_returns_dict, bil,
        date_index=np.arange(n_days),
        top_k_for_combos=10,
        config={"combo_batch_size": 50},
    )

    from math import comb
    expected = comb(10, 2) * 4 * 2   # 360
    assert len(results) == expected, f"Expected {expected} rows, got {len(results)}"


def test_run_combo_backtests_result_keys():
    """Each result dict contains name, member_a, member_b, operator, target, and metrics."""
    rng = np.random.default_rng(1)
    n_days, n_sigs = 100, 5
    sm = rng.random((n_days, n_sigs)) > 0.5
    signal_names = [f"SIG_{i}" for i in range(n_sigs)]
    tr = rng.standard_normal(n_days) * 0.01
    bil = np.full(n_days, 0.0001)

    results = run_combo_backtests(
        sm, signal_names, [None] * n_sigs,
        {"TQQQ": tr}, bil,
        date_index=np.arange(n_days),
        top_k_for_combos=n_sigs,
        config={"combo_batch_size": 100},
    )

    assert len(results) > 0
    row = results[0]
    required_keys = {
        "name", "member_a", "member_b", "operator", "target",
        "total_return", "cagr", "sharpe", "smart_sharpe", "sortino",
        "max_drawdown", "calmar", "omega", "win_rate", "profit_factor",
        "recovery_factor", "time_in_market", "n_signal_days",
    }
    assert required_keys.issubset(set(row.keys())), (
        f"Missing keys: {required_keys - set(row.keys())}"
    )
    # All metric values are plain Python floats
    assert isinstance(row["sharpe"], float)
    assert isinstance(row["name"], str)
    # Name round-trips through parse_combo_name
    members, operators = parse_combo_name(row["name"])
    assert row["member_a"] == members[0]
    assert row["member_b"] == members[1]
    assert row["operator"] == operators[0]


def test_run_combo_backtests_no_cap_when_small():
    """When n_signals <= top_k, all signals are used (no cap applied)."""
    rng = np.random.default_rng(5)
    n_days, n_sigs = 100, 4
    sm = rng.random((n_days, n_sigs)) > 0.5
    signal_names = [f"SIG_{i}" for i in range(n_sigs)]
    tr = rng.standard_normal(n_days) * 0.01
    bil = np.full(n_days, 0.0001)

    results = run_combo_backtests(
        sm, signal_names, [None] * n_sigs,
        {"TQQQ": tr}, bil,
        date_index=np.arange(n_days),
        top_k_for_combos=1000,  # larger than n_sigs
        config={"combo_batch_size": 100},
    )

    from math import comb
    expected = comb(n_sigs, 2) * 4 * 1  # 6 × 4 × 1 target = 24
    assert len(results) == expected
