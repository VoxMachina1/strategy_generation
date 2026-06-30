"""
Tests for src/composer.py — Phase 8: Composer Export.

Covers success criteria:
  SC-1: signal_to_if_child produces correct flat if-child node for all comparators
  SC-2: combo_to_if_child produces correct compound condition for all 4 operators
  SC-3: build_symphony produces correct root/wt-cash-equal/if structure
  SC-4: precond_expr_to_composer_condition parses all supported expression types
  SC-5: verify_composer_output returns match_rate ≥ 0.99 for a known signal
  SC-6: A_AND_NOT_B and B_AND_NOT_A correctly negate the relevant comparator
  SC-7: insert_into_symphony correctly inserts in leaf and root modes
"""

import copy

import numpy as np
import pandas as pd
import pytest

from src.composer import (
    build_symphony,
    combo_to_if_child,
    insert_into_symphony,
    precond_expr_to_composer_condition,
    signal_to_if_child,
    verify_composer_output,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _price_df(n=300, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n, freq="B")
    return pd.DataFrame(
        {"SPY": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
         "QQQ": 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, n))),
         "BIL": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.0001, n)))},
        index=dates,
    )


# ---------------------------------------------------------------------------
# SC-1: signal_to_if_child
# ---------------------------------------------------------------------------

class TestSignalToIfChild:
    def test_basic_structure(self):
        node = signal_to_if_child("RSI_10_SPY_LT_30", "TQQQ")
        assert node["step"] == "if"
        children = node["children"]
        assert len(children) == 2
        true_c, else_c = children
        assert true_c["is-else-condition?"] is False
        assert else_c["is-else-condition?"] is True

    def test_lhs_fields_present(self):
        node = signal_to_if_child("RSI_10_SPY_LT_30", "TQQQ")
        true_c = node["children"][0]
        assert true_c["lhs-fn"] == "relative-strength-index"
        assert true_c["lhs-val"] == "SPY"
        assert true_c["lhs-fn-params"] == {"window": 10}
        assert true_c["comparator"] == "lt"

    def test_fixed_rhs(self):
        node = signal_to_if_child("RSI_10_SPY_LT_30", "TQQQ")
        true_c = node["children"][0]
        assert true_c["rhs-fixed-value?"] is True
        assert true_c["rhs-val"] == "30"

    def test_true_branch_has_target(self):
        node = signal_to_if_child("RSI_14_SPY_GT_70", "TQQQ", safe_asset="BIL")
        true_c = node["children"][0]
        asset = true_c["children"][0]
        assert asset["step"] == "asset"
        assert asset["ticker"] == "TQQQ"

    def test_else_branch_has_safe_asset(self):
        node = signal_to_if_child("RSI_14_SPY_GT_70", "TQQQ", safe_asset="SGOV")
        else_c = node["children"][1]
        asset = else_c["children"][0]
        assert asset["step"] == "asset"
        assert asset["ticker"] == "SGOV"

    def test_sma_fn_name(self):
        node = signal_to_if_child("SMA_20_QQQ_GT_SMA_20_QQQ", "TQQQ")
        true_c = node["children"][0]
        assert true_c["lhs-fn"] == "moving-average-price"

    def test_all_comparators(self):
        for comp, expected in [("GT", "gt"), ("LT", "lt"), ("GTE", "gte"), ("LTE", "lte")]:
            node = signal_to_if_child(f"RSI_10_SPY_{comp}_50", "TQQQ")
            true_c = node["children"][0]
            assert true_c["comparator"] == expected

    def test_gt_comparator_value(self):
        node = signal_to_if_child("RSI_14_SPY_GT_70", "TQQQ")
        true_c = node["children"][0]
        assert true_c["comparator"] == "gt"
        assert true_c["rhs-val"] == "70"


# ---------------------------------------------------------------------------
# SC-2: combo_to_if_child
# ---------------------------------------------------------------------------

class TestComboToIfChild:
    def test_basic_structure(self):
        name = "RSI_10_SPY_LT_30+AND+RSI_10_QQQ_LT_30"
        node = combo_to_if_child(name, "TQQQ")
        assert node["step"] == "if"
        true_c = node["children"][0]
        assert true_c["is-else-condition?"] is False
        assert "condition" in true_c

    def test_and_operator(self):
        name = "RSI_10_SPY_LT_30+AND+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ")
        cond = node["children"][0]["condition"]
        assert cond["condition-type"] == "compound"
        assert cond["operator"] == "all"
        assert len(cond["conditions"]) == 2

    def test_or_operator(self):
        name = "RSI_10_SPY_LT_30+OR+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ")
        cond = node["children"][0]["condition"]
        assert cond["operator"] == "any"

    def test_a_and_not_b_negates_b(self):
        """A_AND_NOT_B: B's comparator must be negated (lt → gte)."""
        name = "RSI_10_SPY_LT_30+A_AND_NOT_B+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ")
        cond = node["children"][0]["condition"]
        assert cond["operator"] == "all"
        a_cond, b_cond = cond["conditions"]
        assert a_cond["comparator"] == "lt"   # A unchanged
        assert b_cond["comparator"] == "gte"  # B negated: lt → gte

    def test_b_and_not_a_negates_a(self):
        """B_AND_NOT_A: A's comparator must be negated."""
        name = "RSI_10_SPY_GT_70+B_AND_NOT_A+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ")
        cond = node["children"][0]["condition"]
        a_cond, b_cond = cond["conditions"]
        assert a_cond["comparator"] == "lte"  # A negated: gt → lte
        assert b_cond["comparator"] == "lt"   # B unchanged

    def test_condition_binary_nodes(self):
        name = "RSI_10_SPY_LT_30+AND+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ")
        cond = node["children"][0]["condition"]
        for sub in cond["conditions"]:
            assert sub["condition-type"] == "binary"
            assert "lhs" in sub and "rhs" in sub

    def test_target_in_true_branch(self):
        name = "RSI_10_SPY_LT_30+AND+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ", safe_asset="BIL")
        true_c = node["children"][0]
        assert true_c["children"][0]["ticker"] == "TQQQ"

    def test_safe_asset_in_else_branch(self):
        name = "RSI_10_SPY_LT_30+AND+RSI_14_QQQ_LT_50"
        node = combo_to_if_child(name, "TQQQ", safe_asset="SGOV")
        else_c = node["children"][1]
        assert else_c["children"][0]["ticker"] == "SGOV"


# ---------------------------------------------------------------------------
# SC-3: build_symphony
# ---------------------------------------------------------------------------

class TestBuildSymphony:
    def _specs(self):
        return [
            {"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"},
            {"signal_name": "RSI_14_SPY_GT_70", "target_ticker": "BIL"},
        ]

    def test_root_structure(self):
        sym = build_symphony(self._specs())
        assert sym["step"] == "root"
        assert sym["rebalance"] == "daily"
        assert len(sym["children"]) == 1
        assert sym["children"][0]["step"] == "wt-cash-equal"

    def test_one_if_block_per_signal(self):
        specs = self._specs()
        sym = build_symphony(specs)
        if_blocks = sym["children"][0]["children"]
        assert len(if_blocks) == len(specs)
        for blk in if_blocks:
            assert blk["step"] == "if"

    def test_combo_signal_uses_compound(self):
        specs = [{"signal_name": "RSI_10_SPY_LT_30+AND+RSI_14_QQQ_LT_50",
                  "target_ticker": "TQQQ"}]
        sym = build_symphony(specs)
        if_block = sym["children"][0]["children"][0]
        true_c = if_block["children"][0]
        assert "condition" in true_c

    def test_base_signal_uses_flat(self):
        specs = [{"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"}]
        sym = build_symphony(specs)
        if_block = sym["children"][0]["children"][0]
        true_c = if_block["children"][0]
        assert "lhs-fn" in true_c   # flat encoding

    def test_precondition_wrapping(self):
        specs = [{
            "signal_name":   "RSI_10_SPY_LT_30",
            "target_ticker": "TQQQ",
            "preconditions": ["RSI('SPY', 14) < 50"],
        }]
        sym = build_symphony(specs)
        outer_if = sym["children"][0]["children"][0]
        # The outer if-child's true branch should contain another if block
        outer_true = outer_if["children"][0]
        assert not outer_true.get("lhs-fn"), "should have compound condition, not flat"
        assert "condition" in outer_true

    def test_empty_specs_returns_empty_symphony(self):
        sym = build_symphony([])
        assert sym["children"][0]["children"] == []


# ---------------------------------------------------------------------------
# SC-4: precond_expr_to_composer_condition
# ---------------------------------------------------------------------------

class TestPrecondExprToComposerCondition:
    def test_rsi_lt_fixed(self):
        cond = precond_expr_to_composer_condition("RSI('SPY', 14) < 50")
        assert cond["condition-type"] == "binary"
        assert cond["comparator"] == "lt"
        assert cond["lhs"]["fn"] == "relative-strength-index"
        assert cond["lhs"]["ticker"] == "SPY"
        assert cond["lhs"]["params"] == {"window": 14}
        assert cond["rhs"] == {"constant": 50.0}

    def test_price_gt_sma(self):
        cond = precond_expr_to_composer_condition("PRICE('SPY') > SMA('SPY', 200)")
        assert cond["comparator"] == "gt"
        assert cond["lhs"]["fn"] == "current-price"
        assert cond["rhs"]["fn"] == "moving-average-price"
        assert cond["rhs"]["params"] == {"window": 200}

    def test_gte_comparator(self):
        cond = precond_expr_to_composer_condition("RSI('QQQ', 10) >= 30")
        assert cond["comparator"] == "gte"

    def test_lte_comparator(self):
        cond = precond_expr_to_composer_condition("RSI('QQQ', 10) <= 70")
        assert cond["comparator"] == "lte"

    def test_unknown_comparator_raises(self):
        with pytest.raises(ValueError):
            precond_expr_to_composer_condition("RSI('SPY', 14) == 50")


# ---------------------------------------------------------------------------
# SC-5: verify_composer_output
# ---------------------------------------------------------------------------

class TestVerifyComposerOutput:
    def test_known_rsi_signal_matches(self):
        """A symphony built from a known RSI signal should match the original matrix."""
        from src.indicators import calculate_rsi
        price_df = _price_df(n=300)

        rsi_vals = calculate_rsi(price_df["SPY"], 10).to_numpy(dtype=float)
        signal_col = np.where(np.isnan(rsi_vals), False, rsi_vals < 30).astype(bool)
        signal_matrix = signal_col.reshape(-1, 1)
        signal_names = ["RSI_10_SPY_LT_30"]

        specs = [{"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"}]
        sym = build_symphony(specs)

        results = verify_composer_output(sym, signal_matrix, signal_names, price_df)
        assert "RSI_10_SPY_LT_30" in results
        r = results["RSI_10_SPY_LT_30"]
        assert r["match_rate"] is not None
        assert r["match_rate"] >= 0.99, f"match_rate={r['match_rate']:.4f}"
        assert r["warning"] is False

    def test_missing_signal_returns_warning(self):
        """If a signal name is not found in the symphony, warning=True."""
        price_df = _price_df(n=100)
        signal_matrix = np.zeros((100, 1), dtype=bool)
        signal_names = ["RSI_99_SPY_LT_30"]  # not in symphony

        specs = [{"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"}]
        sym = build_symphony(specs)

        results = verify_composer_output(sym, signal_matrix, signal_names, price_df)
        assert results["RSI_99_SPY_LT_30"]["warning"] is True

    def test_multiple_signals(self):
        """Multiple signals are each verified independently."""
        from src.indicators import calculate_rsi
        price_df = _price_df(n=200)

        rsi10 = calculate_rsi(price_df["SPY"], 10).to_numpy(dtype=float)
        rsi14 = calculate_rsi(price_df["SPY"], 14).to_numpy(dtype=float)
        col0 = np.where(np.isnan(rsi10), False, rsi10 < 30).astype(bool)
        col1 = np.where(np.isnan(rsi14), False, rsi14 > 70).astype(bool)
        signal_matrix = np.column_stack([col0, col1])
        signal_names = ["RSI_10_SPY_LT_30", "RSI_14_SPY_GT_70"]

        specs = [
            {"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"},
            {"signal_name": "RSI_14_SPY_GT_70",  "target_ticker": "BIL"},
        ]
        sym = build_symphony(specs)
        results = verify_composer_output(sym, signal_matrix, signal_names, price_df)

        assert len(results) == 2
        for name in signal_names:
            assert results[name]["match_rate"] >= 0.99


# ---------------------------------------------------------------------------
# SC-7: insert_into_symphony — Mode C
# ---------------------------------------------------------------------------

def _two_signal_symphony():
    """A minimal symphony with two if-blocks inside wt-cash-equal."""
    return build_symphony([
        {"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"},
        {"signal_name": "RSI_14_SPY_GT_70", "target_ticker": "BIL"},
    ])


def _one_spec():
    return [{"signal_name": "SMA_20_QQQ_GT_SMA_20_QQQ", "target_ticker": "TQQQ"}]


class TestInsertIntoSymphony:
    def test_leaf_appends_to_wt_cash_equal(self):
        sym = _two_signal_symphony()
        result = insert_into_symphony(sym, _one_spec(), mode="leaf")
        wt = result["children"][0]
        assert wt["step"] == "wt-cash-equal"
        assert len(wt["children"]) == 3

    def test_leaf_preserves_existing_children(self):
        sym = _two_signal_symphony()
        original_children = copy.deepcopy(sym["children"][0]["children"])
        result = insert_into_symphony(sym, _one_spec(), mode="leaf")
        kept = result["children"][0]["children"][:2]
        assert kept == original_children

    def test_root_wraps_wt_cash_equal_contents(self):
        sym = _two_signal_symphony()
        original_if_blocks = copy.deepcopy(sym["children"][0]["children"])
        result = insert_into_symphony(sym, _one_spec(), mode="root")
        # Outer wt-cash-equal has one if-block (the gate)
        outer_wt = result["children"][0]
        assert outer_wt["step"] == "wt-cash-equal"
        assert len(outer_wt["children"]) == 1
        gate = outer_wt["children"][0]
        assert gate["step"] == "if"
        # True-child's children should be a wt-cash-equal with the original if-blocks
        true_child = gate["children"][0]
        inner_wt = true_child["children"][0]
        assert inner_wt["step"] == "wt-cash-equal"
        assert inner_wt["children"] == original_if_blocks

    def test_root_multiple_specs_nesting(self):
        sym = _two_signal_symphony()
        original_if_blocks = copy.deepcopy(sym["children"][0]["children"])
        specs = [
            {"signal_name": "RSI_10_SPY_LT_30", "target_ticker": "TQQQ"},
            {"signal_name": "RSI_14_QQQ_LT_50", "target_ticker": "TQQQ"},
        ]
        result = insert_into_symphony(sym, specs, mode="root")
        # Navigate to innermost: root→wt→if→true-child→wt→if→true-child→wt
        outer_if = result["children"][0]["children"][0]
        inner_wt = outer_if["children"][0]["children"][0]
        assert inner_wt["step"] == "wt-cash-equal"
        inner_if = inner_wt["children"][0]
        innermost_wt = inner_if["children"][0]["children"][0]
        assert innermost_wt["step"] == "wt-cash-equal"
        assert innermost_wt["children"] == original_if_blocks

    def test_insert_does_not_mutate_original(self):
        sym = _two_signal_symphony()
        original = copy.deepcopy(sym)
        insert_into_symphony(sym, _one_spec(), mode="leaf")
        assert sym == original
        insert_into_symphony(sym, _one_spec(), mode="root")
        assert sym == original

    def test_invalid_mode_raises(self):
        sym = _two_signal_symphony()
        with pytest.raises(ValueError, match="mode must be"):
            insert_into_symphony(sym, _one_spec(), mode="diagonal")

    def test_leaf_no_wt_cash_equal_raises(self):
        malformed = {"step": "root", "rebalance": "daily", "children": [
            {"step": "if", "children": []}
        ]}
        with pytest.raises(ValueError, match="wt-cash-equal"):
            insert_into_symphony(malformed, _one_spec(), mode="leaf")
