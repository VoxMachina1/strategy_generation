"""
Composer JSON export layer for the Composer Signal Pipeline.

Converts pipeline signals and combos into Composer-compatible JSON symphonies.
Supports two condition encoding formats:
  - Flat if-child (simple signals): comparator/lhs-fn/lhs-val/rhs-val fields on the node
  - Compound condition (combos): nested "condition" object with condition-type/operator/conditions

See crescendo/src/composer/symphony/ComposerSymphonyJson.java for the authoritative schema.

Public API
----------
signal_to_if_child()         — single signal → Composer if-child node (flat encoding)
combo_to_if_child()          — combo signal → Composer if-child node (compound encoding)
build_symphony()             — top-N signals → complete Composer symphony JSON
verify_composer_output()     — round-trip verification against the original signal matrix
"""

import re

import numpy as np

from src.combos import parse_combo_name
from src.signals import parse_signal_name


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Python indicator labels → Composer fn strings (canonical from §8.2)
_FN_MAP: dict[str, str] = {
    "RSI":      "relative-strength-index",
    "SMA":      "moving-average-price",
    "EMA":      "exponential-moving-average-price",
    "CUMRET":   "cumulative-return",
    "MAXDD":    "max-drawdown",
    "MARETURN": "moving-average-return",
    "PRICE":    "current-price",
}

# Negated comparators — used when constructing A_AND_NOT_B / B_AND_NOT_A
_COMP_NEGATE: dict[str, str] = {
    "gt": "lte",
    "lt": "gte",
    "gte": "lt",
    "lte": "gt",
}

# Combo operator → Composer compound junction operator
_OP_JUNCTION: dict[str, str] = {
    "AND":         "all",
    "OR":          "any",
    "A_AND_NOT_B": "all",  # A AND (NOT B) — B comparator gets negated
    "B_AND_NOT_A": "all",  # (NOT A) AND B — A comparator gets negated
}


# ---------------------------------------------------------------------------
# 8.2  Binary condition helpers
# ---------------------------------------------------------------------------

def _fn_raw(label: str) -> str:
    """Resolve a Python indicator label to the Composer fn string."""
    key = label.upper()
    if key not in _FN_MAP:
        raise ValueError(
            f"Unknown indicator label: {label!r}. "
            f"Known labels: {list(_FN_MAP.keys())}"
        )
    return _FN_MAP[key]


def _lhs_expression(parsed: dict) -> dict:
    """
    Build a Composer lhs expression map from a parsed signal dict.

    For compound-condition encoding (binary nodes inside a compound).
    """
    fn = _fn_raw(parsed["lhs_fn"])
    node: dict = {"fn": fn, "ticker": parsed["lhs_ticker"]}
    if parsed.get("lhs_window"):
        node["params"] = {"window": parsed["lhs_window"]}
    return node


def _rhs_expression(parsed: dict, negate: bool = False) -> dict:
    """
    Build a Composer rhs expression map from a parsed signal dict.

    For compound-condition encoding. negate=True is not applied here —
    comparator negation is handled at the binary-condition level.
    """
    if parsed["rhs_type"] == "fixed":
        return {"constant": parsed["rhs_value"]}
    # indicator RHS
    fn = _fn_raw(parsed["rhs_fn"])
    node: dict = {"fn": fn, "ticker": parsed["rhs_ticker"]}
    if parsed.get("rhs_window"):
        node["params"] = {"window": parsed["rhs_window"]}
    return node


def _binary_condition(parsed: dict, negate_comparator: bool = False) -> dict:
    """
    Build a Composer binary condition node (for use inside a compound).

    Parameters
    ----------
    parsed            : output of parse_signal_name()
    negate_comparator : if True, flip the comparator (used for NOT-B patterns)

    Returns
    -------
    dict with keys: condition-type, comparator, lhs, rhs
    """
    comp = parsed["comparator"]
    if negate_comparator:
        comp = _COMP_NEGATE.get(comp, comp)
    return {
        "condition-type": "binary",
        "comparator":     comp,
        "lhs":            _lhs_expression(parsed),
        "rhs":            _rhs_expression(parsed),
    }


# ---------------------------------------------------------------------------
# 8.3  Flat if-child (simple signal)
# ---------------------------------------------------------------------------

def _flat_if_child_fields(parsed: dict, node: dict) -> None:
    """
    Populate flat condition fields on an if-child node (in-place).
    Used for single-signal if-children where no compound is needed.

    Flat format fields:
        comparator, lhs-fn, lhs-val, lhs-fn-params (if window present),
        rhs-fixed-value?, rhs-val / rhs-fn / rhs-fn-params / rhs-fn-lhs-val
    """
    node["comparator"] = parsed["comparator"]

    # LHS
    node["lhs-fn"]  = _fn_raw(parsed["lhs_fn"])
    node["lhs-val"] = parsed["lhs_ticker"]
    if parsed.get("lhs_window"):
        node["lhs-fn-params"] = {"window": parsed["lhs_window"]}

    # RHS
    if parsed["rhs_type"] == "fixed":
        node["rhs-fixed-value?"] = True
        node["rhs-val"]          = parsed["rhs_value"]
    else:
        node["rhs-fixed-value?"]    = False
        node["rhs-fn"]              = _fn_raw(parsed["rhs_fn"])
        node["rhs-fn-lhs-val"]      = parsed["rhs_ticker"]
        if parsed.get("rhs_window"):
            node["rhs-fn-params"]   = {"window": parsed["rhs_window"]}


def signal_to_if_child(
    signal_name: str,
    target_ticker: str,
    safe_asset: str = "BIL",
) -> dict:
    """
    Convert a base signal name to a Composer if-child node (flat encoding).

    The true-branch holds target_ticker; the else-branch holds safe_asset.

    Parameters
    ----------
    signal_name   : canonical signal name, e.g. "RSI_10_SPY_LT_30"
    target_ticker : ticker to hold when signal fires
    safe_asset    : ticker to hold when signal is off (default "BIL")

    Returns
    -------
    Composer if-child dict with flat condition fields, true-branch, and else-branch.
    """
    parsed = parse_signal_name(signal_name)

    true_child: dict = {
        "step":              "if-child",
        "collapsed?":        False,
        "is-else-condition?": False,
    }
    _flat_if_child_fields(parsed, true_child)
    true_child["children"] = [_asset_node(target_ticker)]

    else_child: dict = {
        "step":              "if-child",
        "collapsed?":        False,
        "is-else-condition?": True,
        "children":          [_asset_node(safe_asset)],
    }

    return {
        "step":     "if",
        "children": [true_child, else_child],
    }


# ---------------------------------------------------------------------------
# 8.4  Compound if-child (combo signal)
# ---------------------------------------------------------------------------

def combo_to_if_child(
    combo_name: str,
    target_ticker: str,
    safe_asset: str = "BIL",
) -> dict:
    """
    Convert a combo signal name to a Composer if-child node (compound encoding).

    Supported operators:
      AND         → compound{all: [A, B]}
      OR          → compound{any: [A, B]}
      A_AND_NOT_B → compound{all: [A, negated(B)]}
      B_AND_NOT_A → compound{all: [negated(A), B]}

    Parameters
    ----------
    combo_name    : e.g. "RSI_10_SPY_LT_30+AND+SMA_20_QQQ_GT_SMA_20_TLT"
    target_ticker : ticker to hold when signal fires
    safe_asset    : ticker to hold when signal is off

    Returns
    -------
    Composer if-child dict with a nested "condition" compound object.
    """
    members, operators = parse_combo_name(combo_name)
    if len(operators) != 1:
        raise ValueError(
            f"combo_to_if_child only supports 2-member combos; got {len(operators)} operators"
        )

    op = operators[0]
    parsed_a = parse_signal_name(members[0])
    parsed_b = parse_signal_name(members[1])

    negate_a = op == "B_AND_NOT_A"
    negate_b = op == "A_AND_NOT_B"

    condition: dict = {
        "condition-type": "compound",
        "operator":       _OP_JUNCTION[op],
        "conditions": [
            _binary_condition(parsed_a, negate_comparator=negate_a),
            _binary_condition(parsed_b, negate_comparator=negate_b),
        ],
    }

    true_child: dict = {
        "step":              "if-child",
        "collapsed?":        False,
        "is-else-condition?": False,
        "condition":          condition,
        "children":           [_asset_node(target_ticker)],
    }

    else_child: dict = {
        "step":              "if-child",
        "collapsed?":        False,
        "is-else-condition?": True,
        "children":          [_asset_node(safe_asset)],
    }

    return {
        "step":     "if",
        "children": [true_child, else_child],
    }


# ---------------------------------------------------------------------------
# 8.5  Precondition expression parser
# ---------------------------------------------------------------------------

# Grammar: PRICE('X') op VALUE  |  FN('X', window) op VALUE
# e.g. "PRICE('SPY') > SMA('SPY', 200)"  or  "RSI('SPY', 14) < 70"
_PRECOND_RE = re.compile(
    r"""
    (\w+)               # function name (group 1)
    \(\s*
      '[^']*'|"[^"]*"   # first arg: ticker (ignored for parsing, captured below)
    \s*
    (?:,\s*(\d+))?      # optional window (group 2)
    \s*\)
    \s*
    ([><=!]+)            # comparator (group 3)
    \s*
    (.+)$               # RHS (group 4) — may be function call or literal
    """,
    re.VERBOSE,
)

_COMP_SYMBOLS: dict[str, str] = {
    ">":  "gt",
    "<":  "lt",
    ">=": "gte",
    "<=": "lte",
}


def _parse_precond_side(expr_str: str) -> dict:
    """
    Parse a single side of a precondition expression into a condition-side dict.

    Returns {"type": "fixed", "value": float} or
            {"fn": ..., "ticker": ..., "params": ...}
    """
    expr_str = expr_str.strip()
    try:
        return {"constant": float(expr_str)}
    except ValueError:
        pass

    m = re.match(r"(\w+)\(\s*['\"]([^'\"]+)['\"]\s*(?:,\s*(\d+))?\s*\)", expr_str)
    if m:
        fn_label, ticker, window = m.group(1), m.group(2), m.group(3)
        node: dict = {"fn": _fn_raw(fn_label), "ticker": ticker}
        if window:
            node["params"] = {"window": int(window)}
        return node

    raise ValueError(f"Cannot parse precondition expression side: {expr_str!r}")


def precond_expr_to_composer_condition(expr: str) -> dict:
    """
    Parse a precondition expression string into a Composer binary condition node.

    Supports expressions of the form:
      PRICE('SPY') > SMA('SPY', 200)
      RSI('SPY', 14) < 70
      EMA('QQQ', 50) >= EMA('QQQ', 200)

    Parameters
    ----------
    expr : precondition string in the controlled grammar

    Returns
    -------
    dict with condition-type="binary", comparator, lhs, rhs

    Raises ValueError for unrecognized syntax.
    """
    # Split on comparator token — try two-char first, then one-char
    for comp_sym, comp_key in sorted(_COMP_SYMBOLS.items(), key=lambda x: -len(x[0])):
        if comp_sym in expr:
            lhs_str, _, rhs_str = expr.partition(comp_sym)
            return {
                "condition-type": "binary",
                "comparator":     comp_key,
                "lhs":            _parse_precond_side(lhs_str),
                "rhs":            _parse_precond_side(rhs_str),
            }

    raise ValueError(f"No recognised comparator in precondition: {expr!r}")


# ---------------------------------------------------------------------------
# 8.6  Symphony assembler
# ---------------------------------------------------------------------------

def _asset_node(ticker: str) -> dict:
    """Build a Composer asset node."""
    return {"step": "asset", "ticker": ticker}


def build_symphony(
    top_n_specs: list,
    safe_asset: str = "BIL",
) -> dict:
    """
    Assemble a complete Composer symphony JSON from top-N signal specs.

    Each signal gets one if-block inside a wt-cash-equal weighting node.
    The if-block's true branch holds the signal's target ticker; the else
    branch holds safe_asset.

    Parameters
    ----------
    top_n_specs : list of dicts, each with keys:
                    signal_name  : str — canonical signal or combo name
                    target_ticker: str — ticker to hold when signal fires
                    preconditions: list[str] — optional precondition expressions
    safe_asset  : ticker for the else (off-signal) branch (default "BIL")

    Returns
    -------
    dict — the complete Composer symphony JSON, ready to serialise with json.dumps()
    """
    if_blocks = []

    for spec in top_n_specs:
        signal_name   = spec["signal_name"]
        target_ticker = spec["target_ticker"]
        preconditions = spec.get("preconditions") or []

        # Determine whether this is a combo or a base signal
        try:
            parse_combo_name(signal_name)
            is_combo = True
        except ValueError:
            is_combo = False

        if is_combo:
            if_node = combo_to_if_child(signal_name, target_ticker, safe_asset)
        else:
            if_node = signal_to_if_child(signal_name, target_ticker, safe_asset)

        # Wrap with precondition if present
        if preconditions:
            if_node = _wrap_with_preconditions(if_node, preconditions, safe_asset)

        if_blocks.append(if_node)

    return {
        "step":      "root",
        "rebalance": "daily",
        "children": [
            {
                "step":     "wt-cash-equal",
                "children": if_blocks,
            }
        ],
    }


def _wrap_with_preconditions(
    inner_if: dict,
    preconditions: list,
    safe_asset: str,
) -> dict:
    """
    Wrap an existing if-block inside an outer precondition compound if-block.

    The outer true-branch contains the inner signal if-block;
    the outer else-branch goes to safe_asset.

    For a single precondition, a binary condition is used.
    For multiple preconditions, a compound{all} is used (AND semantics).
    """
    cond_nodes = [precond_expr_to_composer_condition(p) for p in preconditions]

    if len(cond_nodes) == 1:
        condition = cond_nodes[0]
    else:
        condition = {
            "condition-type": "compound",
            "operator":       "all",
            "conditions":     cond_nodes,
        }

    true_child: dict = {
        "step":              "if-child",
        "collapsed?":        False,
        "is-else-condition?": False,
        "condition":          condition,
        "children":           [inner_if],
    }
    else_child: dict = {
        "step":              "if-child",
        "collapsed?":        False,
        "is-else-condition?": True,
        "children":          [_asset_node(safe_asset)],
    }
    return {
        "step":     "if",
        "children": [true_child, else_child],
    }


# ---------------------------------------------------------------------------
# 8.7  Round-trip verification
# ---------------------------------------------------------------------------

def verify_composer_output(
    symphony_json: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    price_df,
) -> dict:
    """
    Re-evaluate the conditions in a generated symphony against historical price data
    and compare them to the original signal matrix.

    Only verifies base signal conditions (flat if-child encoding) where the
    indicator can be computed from `price_df`. Combos (compound encoding) and
    preconditions are not re-evaluated.

    Parameters
    ----------
    symphony_json  : output of build_symphony()
    signal_matrix  : (n_days, n_signals) bool — original signal columns
    signal_names   : list[str] — signal names, parallel to signal_matrix columns
    price_df       : pd.DataFrame with ticker columns and DatetimeIndex

    Returns
    -------
    dict[str, dict] mapping signal_name → {"match_rate": float, "warning": bool}
    """
    from src.indicators import calculate_rsi, calculate_sma, calculate_ema

    _indicator_fns = {
        "relative-strength-index":          calculate_rsi,
        "moving-average-price":             calculate_sma,
        "exponential-moving-average-price": calculate_ema,
    }

    def _eval_flat_if_child(node: dict) -> np.ndarray | None:
        """
        Re-evaluate a flat if-child condition.
        Returns a bool array or None if the node cannot be evaluated.
        """
        fn_raw  = node.get("lhs-fn")
        ticker  = node.get("lhs-val")
        comparator = node.get("comparator")

        if not (fn_raw and ticker and comparator):
            return None
        if fn_raw not in _indicator_fns:
            return None
        if ticker not in price_df.columns:
            return None

        fn_params = node.get("lhs-fn-params") or {}
        window = fn_params.get("window")
        if window is None:
            return None

        series = price_df[ticker]
        ind_vals = _indicator_fns[fn_raw](series, window).to_numpy(dtype=float)

        is_fixed = node.get("rhs-fixed-value?", False)
        if not is_fixed:
            return None  # indicator-vs-indicator RHS not re-evaluated here
        rhs_val = float(node.get("rhs-val", 0))

        nan_mask = np.isnan(ind_vals)
        if comparator == "gt":
            result = ind_vals > rhs_val
        elif comparator == "lt":
            result = ind_vals < rhs_val
        elif comparator == "gte":
            result = ind_vals >= rhs_val
        elif comparator == "lte":
            result = ind_vals <= rhs_val
        else:
            return None

        return np.where(nan_mask, False, result).astype(bool)

    def _find_flat_if_children(node: dict, out: list) -> None:
        """Recursively collect flat-encoded if-child nodes."""
        step = node.get("step")
        if step == "if-child" and not node.get("is-else-condition?", False):
            if node.get("lhs-fn"):
                out.append(node)
        for child in node.get("children", []):
            if isinstance(child, dict):
                _find_flat_if_children(child, out)

    # Collect all flat-encoded true if-child nodes from the symphony
    flat_nodes: list[dict] = []
    _find_flat_if_children(symphony_json, flat_nodes)

    results = {}

    for i, sig_name in enumerate(signal_names):
        original = signal_matrix[:, i]

        # Find the corresponding node by lhs-fn/lhs-val/lhs-fn-params/comparator/rhs-val
        parsed = parse_signal_name(sig_name)
        target_fn  = _FN_MAP.get(parsed["lhs_fn"].upper(), "")
        target_val = parsed["lhs_ticker"]
        target_comp = parsed["comparator"]
        target_rhs  = parsed.get("rhs_value")
        target_win  = parsed.get("lhs_window")

        match_node = None
        for node in flat_nodes:
            if (
                node.get("lhs-fn") == target_fn
                and node.get("lhs-val") == target_val
                and node.get("comparator") == target_comp
                and node.get("rhs-fixed-value?", False)
                and node.get("rhs-val") == target_rhs
                and (node.get("lhs-fn-params") or {}).get("window") == target_win
            ):
                match_node = node
                break

        if match_node is None:
            results[sig_name] = {"match_rate": None, "warning": True}
            continue

        re_evaluated = _eval_flat_if_child(match_node)
        if re_evaluated is None:
            results[sig_name] = {"match_rate": None, "warning": True}
            continue

        match_rate = float(np.mean(re_evaluated == original))
        results[sig_name] = {
            "match_rate": match_rate,
            "warning":    match_rate < 0.99,
        }

    return results
