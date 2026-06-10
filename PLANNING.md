# Composer Signal Pipeline — Planning Document

**Project root:** `signal_pipeline/`  
**Constituent existing projects:** `composer_signal_generator/`, `monte_carlo_sim/`, `rsi_tester/`  
**Status:** Pre-development. Full architectural rewrite of signal_generator. MC sim and RSI search are inputs/references.  
**Last updated:** 2026-06-09  
**Vision document:** See `VISION.md` for high-level goals, signal type architecture, and integration mode design.

---

## Vision

A complete signal discovery and validation pipeline that takes a list of tickers and produces, at the end of a run:

1. A ranked, filtered dataset of signal/target pairs with full out-of-sample performance metrics
2. Monte Carlo reports for the top-N individual signals and the combined portfolio
3. A copy-paste-ready Composer symphony JSON implementing the top-N signals as a strategy

The pipeline handles everything: data download, indicator computation, signal generation, backtesting, walk-forward validation, combo discovery, portfolio construction, Monte Carlo validation, and Composer export.

Two entry points exist:
- **Full pipeline** (`main.py`): exhaustive signal discovery across all indicator types, windows, and ticker combinations. For serious analysis.
- **RSI Search** (`rsi_search.py`): simple mean reversion scan — RSI at various levels across a set of tickers, fast output, no combos. For quick hypothesis testing.

A third entry mode — **Extend Existing Strategy (Mode C)** — is the primary implementation target. See `VISION.md` § "Three Integration Modes" and Phase 8 below.

---

## Signal Type Architecture

Signals are classified into two types based on their regime duration. No user input required — the pipeline auto-classifies after Stage 1 discovery.

### Type 1 — Replacement Signal
- **Median regime duration**: < 20 trading days (configurable)
- **Logic**: fires on bad benchmark days, holds a better asset
- **Assembly**: leaf-node insertion — inserts an `if` block at the leaf node of the target strategy
- **Primary filter**: `benchmark_median_return < 0` (non-negotiable)

### Type 2 — Regime / Timing Signal
- **Median regime duration**: ≥ 20 trading days (configurable)
- **Logic**: portfolio-level gate — long when on, cash (SGOV) when off
- **Assembly**: root-level wrapping — wraps the entire strategy in a new `if` block
- **Primary filters**: `HitRate_Positive_Sharpe`, crisis hold-out Sharpe, stripped Sharpe
- **Note**: regime signals have historically been difficult to make work OOS. The evaluation suite's consistency checks (see Phase 5) are the primary defence.

### Regime Duration Threshold
Default: 20 trading days. Configurable in `config.py`. The threshold separates "day-level replacement" behaviour from "portfolio-level on/off" behaviour. If unsure, inspect the distribution of median regime lengths in the results CSV before interpreting Type assignments.

---

## Current State Assessment

### Existing Code Inventory

| File | Status | Role |
|---|---|---|
| `composer_signal_generator/main.py` | Has bugs, severe performance issues | Signal generation + backtesting + report (5070 lines) |
| `composer_signal_generator/main2.py` | More correct version, same issues | Same as above with precond_mask support (5524 lines) |
| `composer_signal_generator/analysis_workshop.py` | Working, clean | Post-processing filter tool |
| `monte_carlo_sim/Monte Carlo walk forward composer working.py` | Working | Standalone MC simulation |
| `rsi_tester/strategy_engine/src/data_loader.py` | Working | Tiingo downloader (same as fuzz_tester's) |
| `rsi_tester/strategy_engine/src/*.py` (other files) | Empty stubs | Never implemented |

**Use `main2.py` as the reference implementation.** It is more correct than `main.py` (proper `precond_mask` threading through the backtest pipeline). Both are treated as reference only — the rewrite starts clean.

### Back-Ports Required Before Rewrite (main.py only)

Two items from `main2.py` are worth back-porting into `main.py` before the rewrite begins:

1. **`precond_mask` threading** — `main2.py` passes the precondition mask through the runner functions correctly. `main.py` has subtle threading gaps. Back-port the corrected pattern.
2. **`max_workers` user configuration** — `main2.py` increased worker counts. Rather than hardcoding, expose `max_workers` in the user-interactive prompts (or config). Users on low-RAM systems need to keep this low to avoid OOM crashes.

After back-porting, **delete `main2.py`**. It has a confirmed bug (expanding window writes to the rolling output directory) and its other changes are either back-ported or regressions.

### Root Cause Analysis of Current Issues

**Out-of-memory crashes (primary):**  
`backtest_signals()` sends the full price DataFrame and return DataFrame as arguments to every task submitted to the process pool. No initializer pattern is used. With 90,000+ signal tasks, each carrying ~600KB of serialised data, this generates ~54GB of IPC traffic on a 32GB system. The combo workers correctly use the initializer pattern; the base signal workers do not. This is the single biggest change.

**Extreme slowness (primary):**  
Signals are stored as a `dict[str, pd.Series]` and backtested one at a time in Python loops. With 30,000+ signals, this means 30,000 Python iterations per window per method per ticker. The fix is to represent all signals as a single boolean NumPy matrix `(n_days, n_signals)` and compute all backtests simultaneously via matrix operations.

**Results don't match Composer (primary):**  
Two confirmed bugs:
1. `convert_signal_to_composer_format()` splits signal names on `+` without distinguishing signal tokens from operator tokens (`AND`, `OR`, etc.), producing malformed Composer DSL.
2. Preconditions are applied during backtesting but never written into the generated Composer JSON. A strategy backtested with `PRICE('SPY') > SMA('SPY', 200)` as a precondition produces Composer code that has no such filter — the two strategies are structurally different.

**Execution timing note — MOC is VERIFIED CORRECT:**  
`EXECUTION_MODE = "MOC"` (signal at day t → return from day t to day t+1) is the correct simulation of Composer's execution model. Composer evaluates live price data at 3:50PM and trades at the 4PM close. The backtest correctly uses same-day close to compute the signal and next-day close to compute the return. Do not change this. Preserve `EXECUTION_MODE = "MOC"` in the rewrite.

**Combo explosion:**  
The current code generates signals first, then passes all signals into the combo engine with no pre-filtering. The combo generator selects K=30 primaries and M=40 random partners per ticker per window, producing ~1.44 million combo backtests over a full walk-forward run. Per user requirement, the full signal space should be run before quality filtering — quality gates apply at output time, not before combo generation. The fix is making the backtesting of combos fast enough to handle this volume, not reducing the combo count.

**Dead code and structural debt:**  
- `_is_combo_row` defined three times with inconsistent logic
- `_erc_weights()` and `_smart_sharpe_opt()` (non-robust versions) never called
- `_write_combo_distribution_files()` is a 500-line god function with 7 nesting levels running 5–6 redundant `groupby` passes on the same data
- `manifest_rows` is a mutable global list that never resets between runs
- Orphaned module-level code block that never executes
- `DIAGNOSTIC H3:` debug prints left in production code paths

---

## New Architecture

### Project Structure

```
signal_pipeline/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # Tiingo download + CSV cache + freshness check
│   │   └── alignment.py        # Multi-asset price DataFrame builder
│   ├── __init__.py
│   ├── indicators.py           # RSI, SMA, EMA, CumRet, MaxDD, MAReturn
│   ├── signals.py              # Signal matrix generation
│   ├── backtest.py             # Vectorized backtesting engine + metrics
│   ├── combos.py               # Combo generation and backtesting
│   ├── validation.py           # Walk-forward, expanding window, rolling window
│   ├── monte_carlo.py          # Monte Carlo simulation (ported from existing)
│   ├── composer.py             # Composer JSON generation (fixed translation)
│   └── report.py               # HTML and CSV output generation
├── config.py                   # Default configuration + user overrides
├── main.py                     # Full pipeline entry point
├── rsi_search.py               # Simple mean reversion entry point
├── analysis_workshop.py        # Post-processing filter (ported from existing)
├── requirements.txt
├── .env.example
└── PLANNING.md
```

### Data Flow

```
User config (tickers, metrics, windows, targets)
    │
    ▼
[Data Layer] — Tiingo CSVs, freshness-checked, cached locally
    │
    ▼
[Indicator Cache] — compute each (ticker, fn, window) once, store as np.ndarray
    │
    ▼
[Signal Matrix] — boolean np.ndarray (n_days × n_signals)
    │             signal_metadata list (parallel to columns)
    ▼
[Combo Matrix] — extend signal_matrix with all AND/OR/gated combinations
    │             combo_metadata list
    ▼
[Validation Engine] — walk-forward / expanding / rolling slices
    │                  vectorized backtest on each slice
    │                  metrics computed in batch (numpy, no quantstats)
    ▼
[Results Aggregation] — per-signal OOS metrics across all windows
    │                    sorted by user-selected metric
    ▼
[Quality Filter] — applied at OUTPUT time, configurable thresholds
    │               top-N selection
    ▼
[Monte Carlo] — per top-N signal + combined portfolio
    │
    ▼
[Composer Export] — Composer symphony JSON (fixed translation)
    │
    ▼
[Report Generator] — HTML dashboard + CSVs + MC reports
```

---

## Phase 1 — Project Bootstrap and Data Layer

### 1.1 Create Project Structure

Create the directory tree above. Initialize `requirements.txt` with:
- `numpy`, `pandas`, `scipy` — core computation
- `requests`, `python-dotenv` — data download
- `matplotlib`, `seaborn` — charting
- `tqdm` — progress bars
- `composer-tools` — Composer DSL generation (optional import, graceful fallback)

No `quantstats` dependency. All metrics will be implemented natively (see Phase 3.3).

### 1.2 Port Data Layer from Fuzz Tester

Copy `strategy_engine/src/config_loader.py`, `data_loader.py`, and `data_alignment.py` from `fuzz_tester/` into `signal_pipeline/src/data/`. These are tested, working implementations. Minor adaptations:

- `config_loader.py`: remove the `config_dict` passthrough (not needed in pipeline). Simplify to just key loading.
- `data_loader.py`: no changes — already has 429 backoff, freshness check, safe ticker naming.
- `alignment.py`: rename from `data_alignment.py` for clarity. Add a `load_multi_ticker_aligned(tickers, data_dir)` function that returns a single aligned price DataFrame with forward-fill, sorted by date.

### 1.3 Port Indicators

Copy `strategy_engine/src/indicators.py` from `fuzz_tester/` and extend:

```python
# Existing (copy from fuzz_tester):
calculate_sma(series, period)      → pd.Series
calculate_ema(series, period)      → pd.Series
calculate_rsi(series, period)      → pd.Series   # Wilder's smoothing
calculate_cumret(series, period)   → pd.Series   # pct_change over window × 100

# New additions:
calculate_maxdd(series, period)    → pd.Series   # rolling peak-to-trough (from fuzz_tester.py)
calculate_mareturn(series, period) → pd.Series   # rolling mean of daily returns (from fuzz_tester.py)
```

All functions return `pd.Series` aligned to the input index. This is the last pandas boundary — above the indicator layer, everything is numpy.

### 1.4 Indicator Cache

Before signal generation, pre-compute all required indicator series in one pass:

```python
def build_indicator_cache(price_df: pd.DataFrame, 
                          required: list[tuple[str, str, int]]) -> dict:
    """
    required: list of (ticker, fn_label, window) tuples
    Returns: dict keyed by (ticker, fn_label, window) → np.ndarray (n_days,)
    """
    cache = {}
    for ticker, fn, window in required:
        series = price_df[ticker]
        indicator_series = compute_indicator(series, fn, window)
        cache[(ticker, fn, window)] = indicator_series.to_numpy(dtype=float)
    return cache
```

`compute_indicator(series, fn, window)` dispatches to the appropriate `calculate_*` function.

The caller (`generate_signal_matrix`) derives `required` from the config before calling this — no indicator is computed twice regardless of how many signals reference the same (ticker, fn, window).

---

## Phase 2 — Signal Matrix Generation

### 2.1 Signal Representation

Replace the `dict[str, pd.Series]` signal store with:

```python
signal_matrix:   np.ndarray[bool]   # shape (n_days, n_signals)
signal_names:    list[str]           # parallel to columns — human-readable key
signal_metadata: list[dict]          # parallel to columns — full signal spec
date_index:      np.ndarray          # shape (n_days,) — dates as datetime64
```

`signal_metadata[i]` contains:
```python
{
    "name":       str,        # e.g. "RSI_10_SPY_GT_50"
    "lhs_ticker": str,
    "lhs_fn":     str,
    "lhs_window": int,
    "comparator": str,        # "gt", "lt", "gte", "lte"
    "rhs_type":   str,        # "fixed" or "indicator"
    "rhs_value":  float|None, # for fixed threshold
    "rhs_ticker": str|None,
    "rhs_fn":     str|None,
    "rhs_window": int|None,
    "target":     str,        # allocation ticker
}
```

### 2.2 Signal Generation

```python
def generate_signal_matrix(config: dict, 
                            price_df: pd.DataFrame,
                            indicator_cache: dict) -> tuple:
    """
    Returns (signal_matrix, signal_names, signal_metadata, date_index)
    """
```

Signal types generated from config:

**RSI vs fixed threshold:**  
For each `(signal_ticker, rsi_window, threshold, comparator, target)`:
- LHS: `indicator_cache[(signal_ticker, "RSI", rsi_window)]`
- RHS: scalar threshold
- Signal: `lhs_values {comparator} threshold` → boolean array

**SMA/EMA cross (triplet — two assets):**  
For each `(lhs_ticker, rhs_ticker, window, comparator, target)`:
- LHS: `indicator_cache[(lhs_ticker, "SMA", window)]`
- RHS: `indicator_cache[(rhs_ticker, "SMA", window)]`
- Signal: `lhs > rhs` → boolean array

**SMA/EMA multi-window cross:**  
For each `(ticker, short_window, long_window, comparator, target)`:
- LHS: `indicator_cache[(ticker, "SMA", short_window)]`
- RHS: `indicator_cache[(ticker, "SMA", long_window)]`
- Signal: `lhs > rhs` → boolean array

Each signal generates one column in `signal_matrix`. NaN days (warmup period) are set to `False`.

Stack all boolean arrays column-wise:
```python
signal_matrix = np.column_stack(signal_arrays).astype(bool)
```

### 2.3 Signal Naming Convention

Signal names must be parseable for Composer export. Use a structured format:

```
{fn}_{window}_{ticker}_{comparator}_{threshold_or_rhs}
```

Examples:
- `RSI_10_SPY_GT_50` — RSI(SPY, 10) > 50
- `SMA_20_QQQ_GT_SMA_20_TLT` — SMA(QQQ, 20) > SMA(TLT, 20)
- `EMA_50_IWM_GT_EMA_200_IWM` — EMA(IWM, 50) > EMA(IWM, 200)

Combo names are constructed from component names with operator tokens:
- `RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT` — both must be true

The Composer export layer parses these names deterministically. The naming convention is the interface contract between signal generation and Composer export.

---

## Phase 3 — Vectorized Backtesting Engine

### 3.1 Core Backtest Operation

The fundamental operation for all backtesting:

```python
def batch_backtest(signal_matrix: np.ndarray,    # (n_days, n_signals) bool
                   target_returns: np.ndarray,    # (n_days,) float — shifted for MOC
                   bil_returns: np.ndarray,        # (n_days,) float
                  ) -> np.ndarray:                # (n_signals, n_metrics) float
    """
    Vectorized backtest of all signals against one target ticker.
    MOC execution: signal[t] * return[t+1] is pre-baked into target_returns via np.roll.
    """
```

**MOC return preparation (done once before any backtest):**
```python
# MOC: signal at t applies to return from t to t+1
target_returns_moc = np.roll(target_returns, -1)
target_returns_moc[-1] = 0.0   # last day has no next-day return
```

This shift is computed once and passed to all backtest calls. It correctly simulates Composer's behaviour: condition evaluated at close t, trade executes at close t to close t+1.

**Core computation:**
```python
# signal_returns[i, j] = return on day i if signal j was active, else 0
signal_returns = signal_matrix * target_returns_moc[:, np.newaxis]  # (n_days, n_signals)

# BIL returns when not invested
bil_component = (~signal_matrix) * bil_returns[:, np.newaxis]

# Total daily P&L per signal
total_daily = signal_returns + bil_component  # (n_days, n_signals)
```

All metrics are then computed from `signal_returns` and `signal_matrix` using pure NumPy. See Phase 3.3.

### 3.2 Process Pool with Initializer Pattern

For walk-forward and other multi-window evaluation, each window processes all signals in a separate subprocess. The large shared data (signal matrix, return arrays) must be loaded into each worker once at startup, not passed with every task.

```python
# Module-level worker globals
_SIGNAL_MATRIX = None
_TARGET_RETURNS = None  # dict[ticker, np.ndarray]
_BIL_RETURNS = None
_DATE_INDEX = None

def _init_worker(signal_matrix, target_returns, bil_returns, date_index):
    global _SIGNAL_MATRIX, _TARGET_RETURNS, _BIL_RETURNS, _DATE_INDEX
    _SIGNAL_MATRIX = signal_matrix
    _TARGET_RETURNS = target_returns
    _BIL_RETURNS = bil_returns
    _DATE_INDEX = date_index

def _backtest_window(window_spec: dict) -> dict:
    """Worker function — receives only the window date bounds, not the data."""
    start_idx, end_idx = window_spec["start_idx"], window_spec["end_idx"]
    sm_slice = _SIGNAL_MATRIX[start_idx:end_idx]
    # ... compute metrics on slice ...

with ProcessPoolExecutor(
    max_workers=os.cpu_count() - 1,
    initializer=_init_worker,
    initargs=(signal_matrix, target_returns, bil_returns, date_index)
) as pool:
    results = list(pool.map(_backtest_window, window_specs))
```

**Memory sizing check:**  
For the benchmark test (15 signal tickers, 4 targets, ~315 base signals, 7,500 days history):
- `signal_matrix`: 7,500 × 315 × 1 byte = ~2.4MB. Negligible.
- With full combo matrix (~790K combos): 7,500 × 790,000 × 1 byte = ~5.9GB. This does NOT fit in memory at once.

**Combo matrix handling:** See Phase 4 — combos are backtested in batches of ~10,000 columns at a time, not all at once. The signal_matrix (base signals only, ~315 columns) fits comfortably and is passed to workers via the initializer.

### 3.3 Metrics (Pure NumPy — No Quantstats)

All metrics computed from `signal_returns` (n_days,) per-signal:

```python
def compute_metrics_batch(signal_returns: np.ndarray,    # (n_days, n_signals)
                           signal_matrix: np.ndarray,     # (n_days, n_signals) bool
                           annual_factor: float = 252.0,
                          ) -> dict[str, np.ndarray]:     # each value (n_signals,)
```

**Implemented metrics (all vectorized across columns):**

| Metric | Formula |
|---|---|
| Total Return | `(1 + r).prod() - 1` per column |
| CAGR | `(1 + total_return)^(252/n_signal_days) - 1` |
| Sharpe Ratio | `mean(r) / std(r) * sqrt(252)` |
| Smart Sharpe | Sharpe corrected for autocorrelation: `sharpe / sqrt(1 + 2*sum(autocorr(r, k) for k in 1..5))` |
| Sortino Ratio | `mean(r) / std(r[r<0]) * sqrt(252)` |
| Calmar Ratio | `cagr / max_drawdown` |
| Omega Ratio | `sum(max(r, 0)) / sum(max(-r, 0))` |
| Win Rate | `(r > 0).mean()` |
| Profit Factor | `sum(r[r>0]) / abs(sum(r[r<0]))` |
| Recovery Factor | `total_return / max_drawdown` |
| Max Drawdown | `max((peak - trough) / peak)` via cumsum of log returns |
| Time in Market | `signal_matrix.mean()` |
| N Signal Days | `signal_matrix.sum()` |
| Consistency Score | (computed at validation layer — see Phase 5) |

**Tail event metrics (per signal, at base parameters):**

| Metric | Formula |
|---|---|
| Tail Concentration | `sum(top_5pct_positive_returns) / sum(all_positive_returns)` |
| Excess Kurtosis | `scipy.stats.kurtosis(r, fisher=True)` |
| Base Win Rate | `(r > 0).mean()` |
| Stripped Win Rate | Win rate after removing top 5% of return days |
| Win Rate Delta | `base_win_rate - stripped_win_rate` |
| Tail Score | `0.45 * tc_score + 0.30 * kurtosis_score + 0.25 * wr_delta_score` |

For the batch case, kurtosis is computed per-column in a loop (scipy does not vectorize across columns natively). This is acceptable since tail metrics are only computed once at base parameters, not across all sweep points.

**Sorting metrics exposed to user:**

| Sort Key | Label | Description |
|---|---|---|
| `smart_sharpe` | Smart Sharpe | Sharpe corrected for autocorrelation |
| `sortino` | Sortino Ratio | Penalises downside volatility only |
| `calmar` | Calmar Ratio | CAGR / Max Drawdown |
| `omega` | Omega Ratio | Probability-weighted gains vs losses |
| `win_rate_oos` | OOS Win Rate | % of OOS test windows that were profitable |
| `profit_factor` | Profit Factor | Gross gains / gross losses |
| `recovery_factor` | Recovery Factor | Total return / Max drawdown |
| `consistency` | Consistency Score | Hit rate across walk-forward windows |

Default sort: `sortino` descending, tiebreak `calmar`.

---

## Phase 4 — Combo Generation and Backtesting

### 4.1 Combo Operators

Four combination operators, identical semantics to existing code:

| Operator | Token | Logic |
|---|---|---|
| AND | `+AND+` | Both signals must be true |
| OR | `+OR+` | Either signal must be true |
| A AND NOT B | `+A_AND_NOT_B+` | Signal A true AND signal B false |
| B AND NOT A | `+B_AND_NOT_A+` | Signal A false AND signal B true |

### 4.2 Combo Generation Strategy

Per user requirement: run the full signal space before quality filtering. Quality gates are applied at output time only.

**Full pairwise expansion:**
For N base signals, generate C(N, 2) × 4 operator combos = O(2N²) combos.

For the benchmark test (~315 base signals): C(315, 2) × 4 = ~197,820 combos.

This is tractable with vectorized operations. Each combo is a boolean column derived from two base signal columns. The combo matrix is NOT stored all at once — it is generated and backtested in batches.

### 4.3 Batched Combo Backtesting

```python
COMBO_BATCH_SIZE = 10_000   # columns per batch — tune based on available RAM

def run_combo_backtests(signal_matrix, signal_names, signal_metadata,
                        target_returns_dict, bil_returns, date_index,
                        window_slices, config):
    """
    Generate and backtest all pairwise combos in batches.
    Never materializes the full combo matrix in memory.
    """
    n = signal_matrix.shape[1]
    pairs = list(itertools.combinations(range(n), 2))
    operators = ["AND", "OR", "A_AND_NOT_B", "B_AND_NOT_A"]
    
    all_combo_results = []
    
    batch = []
    for i, j in pairs:
        for op in operators:
            batch.append((i, j, op))
            if len(batch) == COMBO_BATCH_SIZE:
                results = _backtest_combo_batch(batch, signal_matrix, ...)
                all_combo_results.extend(results)
                batch = []
    if batch:
        results = _backtest_combo_batch(batch, signal_matrix, ...)
        all_combo_results.extend(results)
    
    return all_combo_results

def _backtest_combo_batch(batch, signal_matrix, target_returns, bil_returns, window_slices):
    """
    Materializes one batch of combo columns, runs backtest, releases memory.
    batch: list of (i, j, op) tuples
    """
    # Build batch combo matrix
    combo_cols = []
    for i, j, op in batch:
        a, b = signal_matrix[:, i], signal_matrix[:, j]
        if op == "AND":            combo_cols.append(a & b)
        elif op == "OR":           combo_cols.append(a | b)
        elif op == "A_AND_NOT_B":  combo_cols.append(a & ~b)
        elif op == "B_AND_NOT_A":  combo_cols.append(~a & b)
    
    batch_matrix = np.column_stack(combo_cols)  # (n_days, batch_size)
    
    # Run vectorized backtest for this batch
    # ... returns metrics array (batch_size, n_metrics) per target ...
    
    # Release combo matrix immediately
    del batch_matrix
    
    return results
```

**Memory estimate per batch:**  
7,500 days × 10,000 combos × 1 byte = 75MB per batch. Well within limits.

**Parallelization:**  
Batches are independent. Use `ProcessPoolExecutor` with `COMBO_BATCH_SIZE` and worker count tuned to available RAM. With 32GB RAM and 75MB per batch: up to ~400 concurrent batches theoretically; in practice use `max_workers = cpu_count - 1` and let the pool manage throughput.

### 4.4 Combo Metadata

Each combo result carries:
```python
{
    "name":      "RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT",
    "member_a":  "RSI_10_SPY_GT_50",
    "member_b":  "SMA_20_QQQ_GT_SMA_20_TLT",
    "operator":  "AND",
    "target":    "TQQQ",
    # + all metrics from compute_metrics_batch
}
```

---

## Phase 5 — Validation Framework

### 5.1 Window Types

Three evaluation methods, all implemented as slices of the signal matrix and return arrays:

**Walk-Forward:** Train on first T days, test on next P days, slide forward by P days.  
Non-overlapping test windows. The most conservative OOS estimate.

**Expanding Window:** Train on days 1..T, test on days T..T+P. Next iteration: train on 1..T+P, test on T+P..T+2P.  
Training window grows with each iteration. Test windows are non-overlapping.

**Rolling Window:** Train on a fixed-length window, slide forward by step.  
All windows the same size. Tests both the signal's consistency and its recency.

### 5.2 Consistency Score

After all window iterations complete, compute per-signal:
```python
consistency_score = n_windows_positive_sharpe / n_windows_total
```

This is the "hit rate across walk-forward windows" — the most direct measure of whether the edge is durable across time vs concentrated in a few good periods.

### 5.3 Aggregation

For each signal, aggregate across all OOS windows:
- Median OOS Sharpe (Sharpe_p50)
- 10th percentile OOS Sharpe (Sharpe_p10) — worst-case
- 90th percentile OOS Sharpe (Sharpe_p90) — best-case
- OOS Sharpe IQR (Sharpe_IQR) — stability
- Consistency Score / HitRate_Positive_Sharpe (hit rate)
- Sharpe_CoV — coefficient of variation across windows (high CoV = one great period)
- Median OOS Total Return (Return_p50)
- 90th percentile Max Drawdown (MaxDD_p90) — worst-case drawdown

These are the columns written to the output CSV and used for sorting and filtering.

### 5.4 Crisis Hold-Out Filter

A set of fixed crisis epochs is defined in `config.py` (hardcoded defaults, fully user-editable):

```python
CRISIS_EPOCHS = [
    ("2008-09-01", "2009-06-30"),   # GFC
    ("2020-02-01", "2020-05-31"),   # COVID crash
    ("2022-01-01", "2022-12-31"),   # Rate hike bear
]
```

For each signal during aggregation:
- If the signal **never fires** on any day in a crisis epoch → no penalty, no bonus.
- If the signal **fires** during a crisis epoch → compute the Sharpe of signal-day returns within that epoch. Must pass a configurable floor (default `0.0`). A signal that fires into a crash and loses is filtered out.

This catches signals that pass all walk-forward checks purely because every crisis period fell in training windows. It does **not** penalise signals that are simply inactive during crises — inactivity is neutral.

Configuration keys (in `config.py`):
```python
CRISIS_EPOCHS: list[tuple[str, str]]   # start/end date pairs
CRISIS_SHARPE_FLOOR: float = 0.0       # minimum crisis-period Sharpe when signal fires
```

### 5.5 Stripped Sharpe

Compute the aggregate OOS Sharpe with the single best OOS window excluded. Added as a column (`Sharpe_Stripped`) in the results CSV.

- If stripped Sharpe is positive: performance is distributed across windows.
- If stripped Sharpe collapses to near zero or negative: the aggregate is dominated by one episode (one-hit wonder).

This operates at the window level and complements the anti-home-run filter (which caps `Sharpe_p90`, operating at the return-day level).

### 5.6 Regime-Level Analysis

After walk-forward validation, compute per-signal regime statistics from the full signal series:

```python
def compute_regime_stats(signal_col: np.ndarray, target_returns: np.ndarray) -> dict:
    """
    Identifies contiguous 'on' blocks (regime episodes) and computes per-episode stats.
    """
```

**Outputs (added as columns in the results CSV):**

| Column | Description |
|---|---|
| `Regime_Count` | Number of contiguous "on" blocks over the full period |
| `Regime_Duration_Median` | Median length of "on" blocks (trading days) |
| `Regime_Duration_Max` | Longest "on" block |
| `Regime_Hit_Rate` | Fraction of "on" blocks with positive total return |
| `Signal_Type` | `"Type1"` or `"Type2"` (from median duration vs threshold) |

**Type classification rule:**
```python
signal_type = "Type2" if regime_duration_median >= REGIME_TYPE_THRESHOLD else "Type1"
```

Where `REGIME_TYPE_THRESHOLD` defaults to 20 trading days.

---

## Phase 5b — Tail Analysis (Automated)

Tail analysis is currently only in Strategy Viewer (interactive). The pipeline automates it for every validated signal.

The implementation is a direct port of `compute_tail_metrics()` from `fuzz_tester.py`. Compute at the **base parameter point** only (not across sweep cells).

**Inputs:** signal-day return series from the full evaluation period (the actual daily returns on days the signal was active).

**Outputs (added as columns in the results CSV):**

| Column | Description |
|---|---|
| `Tail_Concentration` | % of total profit from top 5% of signal days |
| `Excess_Kurtosis` | Fisher kurtosis of signal-day returns |
| `Base_Win_Rate` | Win rate at base parameters |
| `Stripped_Win_Rate` | Win rate with top 5% of return days removed |
| `WR_Delta` | `Base_Win_Rate - Stripped_Win_Rate` |
| `Tail_Score` | Composite (0–1): `0.45 * tc + 0.30 * ek + 0.25 * wd` |

**Combined rank score (for default output sort):**

```python
combined_score = 0.6 * oos_quality_score + 0.4 * tail_score
```

Where `oos_quality_score` is derived from the walk-forward metrics (normalised Sharpe_p50 × consistency). This is the same composite used in Strategy Viewer's sidebar ranking.

---

## Phase 6 — Monte Carlo Integration

### 6.1 Scope

Monte Carlo is run after validation, on the top-N signals (configurable) plus the combined portfolio. Per user requirement, even signals that didn't make the top-N are included in MC if they cleared a minimal quality bar (configurable minimum win rate or Sharpe) — the output should give the user data to compare.

### 6.2 Porting the Existing MC Sim

Port the core functions from `monte_carlo_sim/Monte Carlo walk forward composer working.py`:

- `run_monte_carlo_simulation(returns, num_simulations, simulation_length)` — unchanged
- `analyze_drawdowns(returns, ...)` — unchanged  
- `run_walk_forward_test(dates, returns, test_period_length, ...)` — unchanged
- `plot_drawdown_distributions(...)` — unchanged

Remove the Composer API integration from the MC sim (that was the `fetch_backtest()` function which hit the Composer backtest API). In the pipeline, the return series come from the validated signal backtests, not from Composer's API.

### 6.3 Per-Signal MC

For each of the top-N signals:
- Extract the OOS return series from the walk-forward validation (the actual realized OOS daily returns, not in-sample)
- Run `run_monte_carlo_simulation` with 10,000 paths
- Run `run_walk_forward_test` for 3-month, 6-month, and 1-year horizons
- Output: PNG charts + summary stats dict

### 6.4 Portfolio MC

Construct equal-weight portfolio from top-N signals:
```python
portfolio_returns = np.mean(np.column_stack(top_n_oos_returns), axis=1)
```

Run the same MC suite on `portfolio_returns`. This is the most important output — the combined strategy's MC distribution tells the user whether the strategy holds up as a whole.

---

## Phase 7 — RSI Search (Simple Mean Reversion Path)

### 7.1 Overview

Separate entry point: `rsi_search.py`. Implements the RSI Search tool described at `github.com/VoxMachina1/rsi_search`. Fast, no combos, no walk-forward — useful for quick hypothesis testing.

### 7.2 Config

```python
{
    "signal_tickers":   ["SPY", "QQQ", "IWM", ...],   # tickers to compute RSI on
    "target_tickers":   ["TQQQ", "SQQQ", "TLT", ...], # tickers to hold when signal fires
    "rsi_windows":      [5, 10, 14, 20],
    "rsi_thresholds":   [20, 30, 40, 50, 60, 70, 80],
    "comparators":      ["lt", "gt"],                  # RSI < threshold or RSI > threshold
    "benchmark_tickers": ["BIL", "SPY"],               # benchmark for win rate comparison
    "min_trades":        20,
    "min_win_rate":      0.75,
}
```

### 7.3 Output

CSV with one row per (signal_ticker, rsi_window, threshold, comparator, target_ticker):

| Column | Description |
|---|---|
| Signal | e.g. `RSI_10_SPY_LT_30` |
| Target | e.g. `TQQQ` |
| Win_Rate | % of signal days where Target beat BIL |
| N_Trades | Signal days in backtest window |
| Benchmark_Median_Return | Median return of BIL on signal days |
| Total_Return | Cumulative return of Target on signal days |
| Sharpe | Annualized Sharpe of Target on signal days |
| Tail_Concentration | % of profit from top 5% of signal days |

Filtered by `min_trades` and `min_win_rate` before output. `benchmark_median_return < 0` is flagged but not filtered by default (user configurable).

### 7.4 "Best Asset at Rebalance" Logic

The RSI search config includes `target_tickers` as a list. For a given signal firing, the tool identifies which target ticker had the best performance on signal days — the "mechanical overfitter" behaviour is now transparent and explicitly labelled as in-sample selection rather than a model output. The output CSV includes a `Best_Target_IS` column (in-sample best) to make this visible, alongside metrics for each individual target.

---

## Phase 8 — Composer Export (Fixed)

### 8.1 Signal Name Parser

The key bug in the existing code: splitting on `+` conflates signal name tokens with operator tokens. Fix: use the known operator set to split the name into members and operators:

```python
OPERATORS = {"AND", "OR", "A_AND_NOT_B", "B_AND_NOT_A"}

def parse_combo_name(signal_name: str) -> tuple[list[str], list[str]]:
    """
    Splits a combo signal name into (members, operators).
    e.g. "RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT"
      → (["RSI_10_SPY_GT_50", "SMA_20_QQQ_GT_SMA_20_TLT"], ["AND"])
    """
    tokens = signal_name.split("+")
    members, ops = [], []
    for tok in tokens:
        if tok in OPERATORS:
            ops.append(tok)
        else:
            members.append(tok)
    if len(members) < 1:
        raise ValueError(f"Could not parse combo name: {signal_name!r}")
    return members, ops

def parse_signal_name(signal_name: str) -> dict:
    """
    Parses a base signal name into its components.
    e.g. "RSI_10_SPY_GT_50" → {lhs_fn, lhs_window, lhs_ticker, comparator, rhs_type, rhs_value}
    """
    # Structured format: {fn}_{window}_{ticker}_{comparator}_{rhs}
    # ...
```

### 8.2 Condition → Composer DSL

Map each parsed signal to a Composer condition node:

```python
def signal_to_composer_condition(parsed: dict, ticker_override: str = None) -> dict:
    """
    Converts a parsed signal spec to a Composer JSON condition node.
    """
    fn_map = {
        "RSI": "relative-strength-index",
        "SMA": "moving-average-price",
        "EMA": "exponential-moving-average-price",
        "CumRet": "cumulative-return",
        "MaxDD": "max-drawdown",
        "MAReturn": "moving-average-return",
        "Price": "current-price",
    }
    # Build lhs, rhs, comparator nodes per Composer schema
    # ...
```

### 8.3 Precondition Translation

Preconditions are currently applied in backtesting but never written to Composer output. This must be fixed:

If the user configured preconditions (e.g. `PRICE('SPY') > SMA('SPY', 200)`), the generated symphony JSON must wrap the signal conditions inside a nested `compound` condition that includes both the precondition and the signal.

The precondition expressions (from the `_safe_eval_precond` grammar) must be parseable back into Composer condition nodes. Since the precondition grammar is a controlled subset (`PRICE`, `SMA`, `EMA`, `RSI` function calls with comparators), a dedicated parser can convert them to Composer condition nodes without `eval`.

### 8.4 Round-Trip Verification

After generating the Composer JSON, verify correctness:
1. Parse the generated JSON using `extract_conditions_from_tree` (from the Strategy Viewer's extractor — identical logic)
2. Re-evaluate the extracted conditions against the historical price data
3. Compare the resulting boolean signal to the original `signal_matrix` column
4. If they disagree on more than 1% of days, log a warning and include a diff summary

This catches translation errors before the user pastes the code into Composer.

### 8.5 Type 2 Assembly — Root-Level Wrapping

Type 2 (regime) signals use a completely different insertion mechanism from Type 1. Instead of navigating to a leaf node, the assembler wraps the entire existing strategy tree in a new `if` block at the root boundary.

**New structure:**
```json
{
  "step": "root",
  "rebalance": "daily",
  "children": [
    {
      "step": "wt-cash-equal",
      "children": [
        {
          "step": "if",
          "children": [
            {
              "step": "if-child",
              "is-else-condition?": false,
              "lhs-fn": "<signal_fn>",
              "lhs-window-days": "<window>",
              "lhs-val": "<ticker>",
              "comparator": "<gt|lt>",
              "rhs-fixed-value?": true,
              "rhs-val": "<threshold>",
              "children": [ "<original root.children[0] — entire existing strategy>" ]
            },
            {
              "step": "if-child",
              "is-else-condition?": true,
              "children": [
                { "step": "asset", "ticker": "SGOV", "exchange": "ARCX" }
              ]
            }
          ]
        }
      ]
    }
  ]
}
```

The `original root.children[0]` node is the existing strategy's first child (typically a `wt-cash-equal` or `group` node) attached verbatim — the entire subtree is untouched.

**Stacking multiple Type 2 signals:** nest wrappers. Each outer signal gates the inner strategy. The innermost `if-child` (true) contains the original strategy; each level's `else` goes to cash.

**Code separation:** Type 1 and Type 2 assembly are separate code paths with no shared logic. The Type 2 path is simpler — it constructs a shallow wrapper around the loaded JSON root, rather than recursively searching for leaf nodes.

### 8.6 Mode C — Extend Existing Strategy

**Entry point:** User supplies a path to a Composer strategy JSON alongside the standard config.

**Pipeline flow:**
1. Load and parse the existing JSON. Extract:
   - All leaf `asset` tickers → candidates for `target` in new signals
   - All `lhs-val` / `rhs-val` tickers used in existing conditions → include in signal ticker pool
   - The user assigns a benchmark from the extracted tickers (or overrides with any ticker)
2. Run Stage 1 discovery normally, using the combined ticker pool.
3. Run Stages 2–5b (validation, regime analysis, tail analysis) normally.
4. At Stage 5 (assembly), route by signal type:
   - **Type 1 signals**: insert at the leaf node of the closest matching path in the existing JSON. If the target asset appears at multiple leaves, insert at each. Falls back to "new JSON" mode if the target is not present in the existing strategy.
   - **Type 2 signals**: wrap the entire existing JSON root.
5. Output the modified JSON alongside a report summarising which signals were inserted, where, and why.

**Design note:** "Wherever I want" insertion is supported by allowing the user to specify an insertion target (leaf asset name, or "root") as an override. Without an override, the assembler infers the insertion point from the signal's target ticker.

### 8.7 Symphony Structure (Type 1 — New Strategy)

For top-N Type 1 signals assembled as a new strategy (not extending an existing one):

```json
{
  "step": "root",
  "rebalance": "daily",
  "children": [
    {
      "step": "wt-cash-equal",
      "children": [
        {
          "step": "if",
          "children": [
            {
              "step": "if-child",
              "is-else-condition?": false,
              "condition": { ... signal condition ... },
              "children": [{ "step": "asset", "ticker": "TQQQ" }]
            },
            {
              "step": "if-child",
              "is-else-condition?": true,
              "children": [{ "step": "asset", "ticker": "BIL" }]
            }
          ]
        }
        // ... one if block per signal in top-N ...
      ]
    }
  ]
}
```

Each signal gets equal weight via `wt-cash-equal`. The safe asset (BIL or user-configured) is held in the else branch.

---

## Phase 9 — Output and Report Generation

### 9.1 CSV Outputs

Written to `output/{run_timestamp}/`:

| File | Contents |
|---|---|
| `all_signals.csv` | Every base signal with all OOS metrics |
| `all_combos.csv` | Every combo with all OOS metrics |
| `top_n_signals.csv` | Top-N filtered and sorted, all metrics |
| `rsi_search.csv` | RSI Search results (if RSI Search path was run) |

### 9.2 Monte Carlo HTML Reports

One HTML file per top-N signal (ported from existing MC sim output), plus one for the combined portfolio. Written to `output/{run_timestamp}/monte_carlo/`.

### 9.3 Composer JSON

`output/{run_timestamp}/symphony.json` — copy-paste ready.

### 9.4 Summary HTML Dashboard

`output/{run_timestamp}/report.html` — a single self-contained HTML file (same philosophy as the Strategy Viewer) showing:

- Run configuration summary
- Top-N signals table with all metrics, sortable by any column
- Tail event analysis per signal (Tail Concentration, Kurtosis, Stripped Win Rate)
- Combined portfolio equity curve vs BIL
- Links to individual MC reports
- Fragility/tail scoring colour-coded badges per signal

### 9.5 Analysis Workshop (Ported)

`analysis_workshop.py` is ported from the existing implementation with no changes to its core logic. It reads CSVs from the `output/` directory and provides interactive post-hoc filtering. The port adds:
- Auto-discovery of the most recent run output directory
- Omega Ratio added to the filter menu (computed on load if not present in CSV)
- Tail Concentration, Excess Kurtosis, Stripped Win Rate added as filterable columns

---

## Known Issues Inherited from Existing Code

| Issue | Severity | Fix in Phase |
|---|---|---|
| Combo signal name parsing broken (splits operators as tokens) | HIGH | Phase 8.1 |
| Preconditions not translated to Composer output | HIGH | Phase 8.3 |
| `_is_combo_row` defined 3 times with inconsistent logic | MEDIUM | Phase 2 (use single function) |
| `_write_combo_distribution_files` god function (500 lines, 7 nesting levels) | MEDIUM | Phase 9 (rewrite as clean aggregator) |
| `manifest_rows` mutable global never resets | LOW | Phase 1 (eliminate) |
| Orphaned module-level code block | LOW | Phase 1 (remove) |
| `DIAGNOSTIC H3:` debug prints in production code | LOW | Phase 1 (remove) |
| `get_user_inputs()` dead function (superseded by `get_enhanced_user_inputs`) | LOW | Phase 1 (remove) |

---

## Performance Targets

**Benchmark test configuration:**  
15 signal tickers × 4 target assets × 3 metrics (RSI, SMA, EMA) × 3 RSI thresholds × 9 SMA/EMA windows (10–50 in steps of 5)

**Estimated signal count:**
- RSI: 15 tickers × 3 thresholds = 45 signal definitions
- SMA cross: 15 tickers × 9 windows × (15 other tickers) × 9 windows / 2 = ~9,112 SMA-vs-SMA pairs (the user's "triplets")
- EMA cross: same as SMA = ~9,112

Total base signals (rough): ~18,000–20,000 before target expansion

Per target: ~18,000 signals × 4 targets = 72,000 signal-target pairs

**Combo count:** C(18,000, 2) × 4 = ~648M combos — this is too large for exhaustive exploration. The combo engine should cap at a configurable maximum (default: top-500 signals by individual OOS Sharpe → C(500, 2) × 4 = ~998K combos). Document this cap clearly in the UI and config.

**Revised combo strategy:** Run individual signal backtests for all ~72,000 signal-target pairs. For combo generation, take the top-K individual signals (configurable, default 500) as the combo pool. This is not a quality gate (the full signal space was explored); it is a combinatorial feasibility cap. The top-K selection is by OOS metric, ensuring the most promising combinations are explored.

**Target runtimes:**
- RSI Search path: under 5 minutes
- Full pipeline (individual signals, no combos): under 30 minutes
- Full pipeline with combos (top-500 pool): under 4 hours

These assume multiprocessing with `cpu_count - 1` workers. Actual timing depends on system specs. Tune `COMBO_BATCH_SIZE` and worker count in `config.py` based on observed performance.

---

## Execution Timing Reference

**MOC mode is VERIFIED CORRECT. Do not change it.**

Composer evaluates live price data at 3:50PM and executes at the 4PM close. The backtest correctly models this as:
- Signal condition evaluated using close price at day t
- Return applied: close at day t → close at day t+1

Implementation: `target_returns_moc = np.roll(target_returns, -1); target_returns_moc[-1] = 0.0`

The `NEXT_BAR` mode (signal at t-1 → return at t) does NOT match Composer's execution model and must not be used as the default.

---

## Success Criteria

The following must all be true before this project is considered complete:

1. `python main.py` runs the benchmark test configuration to completion without OOM crash or exception.
2. Benchmark test completes within the target runtime (individual signals: 30 min; with combos: 4 hours).
3. `output/{timestamp}/symphony.json` is valid Composer JSON that can be imported into Composer.Trade.
4. The round-trip verification (Phase 8.4) passes with < 1% signal disagreement for all top-N signals.
5. `python rsi_search.py` completes in under 5 minutes for a 15-ticker, 4-target config.
6. The combined portfolio Monte Carlo report is generated and readable.
7. `analysis_workshop.py` successfully loads the top-N CSV and all tail event columns are present and filterable.
8. A signal backtested with an active precondition produces a Composer JSON that includes the precondition as a nested condition node.
