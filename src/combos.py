"""
Combo generation and backtesting for the Composer Signal Pipeline.

Combos are boolean combinations of two base signals using one of four operators.
The combo matrix is NEVER materialised in full — combos are generated and
backtested in batches of COMBO_BATCH_SIZE columns, then immediately released.

Public API
----------
parse_combo_name()       — split a combo name into (members, operators)
apply_operator()         — apply one of the four boolean operators
build_combo_batch()      — build a (n_days, batch_size) boolean matrix for a batch
run_combo_backtests()    — full pairwise combo sweep with batched backtesting

Naming convention (matches existing Composer export layer)
----------------------------------------------------------
  member_a + "+" + OPERATOR + "+" + member_b
  e.g. "RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT"

Operators: AND, OR, A_AND_NOT_B, B_AND_NOT_A
"""

import itertools
import re

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from src.backtest import batch_backtest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COMBO_BATCH_SIZE = 10_000   # columns per batch — tune based on available RAM

_OPERATORS = ("AND", "OR", "A_AND_NOT_B", "B_AND_NOT_A")

# Split pattern: +OPERATOR+ — capturing group so split() includes the operator tokens
_OP_PATTERN = re.compile(r"\+(AND|OR|A_AND_NOT_B|B_AND_NOT_A)\+")


# ---------------------------------------------------------------------------
# 4.1 Combo name parsing
# ---------------------------------------------------------------------------

def parse_combo_name(name: str) -> tuple:
    """
    Split a combo signal name into its member signal names and operators.

    Parameters
    ----------
    name : str
        e.g. "RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT"

    Returns
    -------
    (members, operators) where:
        members   : list[str] — base signal names (len = n_operators + 1)
        operators : list[str] — operator tokens from {"AND","OR","A_AND_NOT_B","B_AND_NOT_A"}

    Raises ValueError if the name contains no recognised operator token.

    Note: this is the correct fix for the existing code's bug of splitting on "+"
    indiscriminately, which confuses signal-name underscores with operator tokens.
    Always call this before calling parse_signal_name() on combo members.
    """
    parts = _OP_PATTERN.split(name)
    # re.split with a capturing group returns [member, op, member, op, member, ...]
    members = parts[0::2]
    operators = parts[1::2]
    if not operators:
        raise ValueError(
            f"No recognised operator found in combo name: {name!r}. "
            f"Known operators: {_OPERATORS}"
        )
    return members, operators


def make_combo_name(member_a: str, operator: str, member_b: str) -> str:
    """Build the canonical combo name string."""
    return f"{member_a}+{operator}+{member_b}"


# ---------------------------------------------------------------------------
# 4.2 Operator application and batch construction
# ---------------------------------------------------------------------------

def apply_operator(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    """
    Apply one of the four boolean combination operators to two signal columns.

    Parameters
    ----------
    a, b : np.ndarray of dtype bool, shape (n_days,)
    op   : one of "AND", "OR", "A_AND_NOT_B", "B_AND_NOT_A"

    Returns
    -------
    np.ndarray of dtype bool, shape (n_days,)
    """
    if op == "AND":
        return a & b
    elif op == "OR":
        return a | b
    elif op == "A_AND_NOT_B":
        return a & ~b
    elif op == "B_AND_NOT_A":
        return ~a & b
    else:
        raise ValueError(f"Unknown operator: {op!r}. Must be one of {_OPERATORS}")


def build_combo_batch(
    signal_matrix: np.ndarray,
    batch: list,
) -> np.ndarray:
    """
    Materialise one batch of combo columns from the base signal matrix.

    Parameters
    ----------
    signal_matrix : np.ndarray, shape (n_days, n_signals), dtype bool
    batch         : list of (i: int, j: int, op: str) tuples

    Returns
    -------
    np.ndarray, shape (n_days, batch_size), dtype bool

    Caller is responsible for releasing the returned array after use.
    """
    cols = [
        apply_operator(signal_matrix[:, i], signal_matrix[:, j], op)
        for i, j, op in batch
    ]
    return np.column_stack(cols).astype(bool)


# ---------------------------------------------------------------------------
# 4.3 Batched combo backtester
# ---------------------------------------------------------------------------

def run_combo_backtests(
    signal_matrix: np.ndarray,
    signal_names: list,
    signal_metadata: list,
    target_returns_dict: dict,
    bil_returns: np.ndarray,
    date_index: np.ndarray,
    top_k_for_combos: int = 500,
    config: dict = None,
) -> list:
    """
    Generate and backtest all pairwise combos in batches.

    Never materialises the full combo matrix in memory — each batch of
    COMBO_BATCH_SIZE combo columns is built, backtested, then immediately
    released before the next batch is built.

    Parameters
    ----------
    signal_matrix      : (n_days, n_signals) bool — base signal matrix
    signal_names       : list[str] — signal names, parallel to columns
    signal_metadata    : list[SignalSpec] — metadata, parallel to columns
    target_returns_dict: dict[str, np.ndarray] — ticker → MOC-shifted returns (n_days,)
    bil_returns        : np.ndarray (n_days,) — BIL daily returns
    date_index         : np.ndarray (n_days,) — calendar dates
    top_k_for_combos   : int — pool size for combo generation.
                         NOTE: this is a COMBINATORIAL FEASIBILITY CAP, not a quality
                         filter. It limits C(K,2)×4 to a tractable number of combos.
                         Quality gates are applied at OUTPUT TIME, not here. Default 500.
    config             : dict, optional. Recognised keys:
                           "combo_batch_size" (int, default COMBO_BATCH_SIZE)

    Returns
    -------
    list[dict] — one dict per (combo, target_ticker) pair, containing:
        name, member_a, member_b, operator, target,
        total_return, cagr, sharpe, smart_sharpe, sortino, max_drawdown,
        calmar, omega, win_rate, profit_factor, recovery_factor,
        time_in_market, n_signal_days
    """
    if config is None:
        config = {}
    batch_size = int(config.get("combo_batch_size", COMBO_BATCH_SIZE))

    n_signals = signal_matrix.shape[1]

    # --- Select combo pool (top-K by mean sortino across all targets) -------
    # Combinatorial feasibility cap — NOT a quality filter.
    # With K signals: C(K,2) × 4 combos. At K=500: ~499,000 combos.
    # At K=n_signals (no cap): C(n,2) × 4 combos, potentially millions.
    pool_size = min(top_k_for_combos, n_signals)

    if pool_size < n_signals:
        # Score each signal by mean sortino across all target tickers
        sortino_scores = np.zeros(n_signals)
        for ticker, tr_moc in target_returns_dict.items():
            metrics = batch_backtest(signal_matrix, tr_moc, bil_returns)
            sortino_scores += metrics["sortino"]
        sortino_scores /= max(len(target_returns_dict), 1)
        pool_indices = np.argsort(sortino_scores)[::-1][:pool_size]
    else:
        pool_indices = np.arange(n_signals)

    # Sub-matrix and names for the combo pool
    pool_matrix = signal_matrix[:, pool_indices]
    pool_names = [signal_names[i] for i in pool_indices]

    # --- Generate all (i, j, op) triplets from the pool ---------------------
    all_triplets = [
        (i, j, op)
        for (i, j), op in itertools.product(
            itertools.combinations(range(pool_size), 2),
            _OPERATORS,
        )
    ]

    # --- Batch loop ----------------------------------------------------------
    all_results = []

    for batch_start in range(0, len(all_triplets), batch_size):
        batch = all_triplets[batch_start : batch_start + batch_size]

        # Materialise combo columns for this batch
        combo_matrix = build_combo_batch(pool_matrix, batch)   # (n_days, batch_size)

        # Backtest against every target ticker
        for ticker, tr_moc in target_returns_dict.items():
            metrics = batch_backtest(combo_matrix, tr_moc, bil_returns)

            for k, (i, j, op) in enumerate(batch):
                name_a = pool_names[i]
                name_b = pool_names[j]
                row = {
                    "name":     make_combo_name(name_a, op, name_b),
                    "member_a": name_a,
                    "member_b": name_b,
                    "operator": op,
                    "target":   ticker,
                }
                for metric_key, arr in metrics.items():
                    row[metric_key] = float(arr[k])
                all_results.append(row)

        # Release combo matrix immediately — do not accumulate
        del combo_matrix

    return all_results
