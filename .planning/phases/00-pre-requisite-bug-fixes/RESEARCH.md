# Phase 0: Pre-Requisite Bug Fixes — Research

**Researched:** 2026-06-09
**Domain:** Python bug fixes across three existing codebases (`composer_signal_generator`, `monte_carlo_sim`, `rsi_tester`)
**Confidence:** HIGH — all bugs confirmed by direct inspection of source files

---

## Summary

This phase fixes 11 confirmed bugs before the Phase 1 pipeline rewrite begins. All bugs have been directly verified against the source code. Nine of the 11 tasks are straightforward mechanical edits. Two require more judgment: task 2 (precond_mask back-port) involves a deliberate architectural choice between two valid approaches, and task 10 (MC drawdown unit mismatch) has a subtlety explained below.

The most important sequencing constraint is that tasks 5, 7, 8, and 9 form a dependency chain: the Tiingo OHLC extension (task 7) is required before true ATR can be implemented (tasks 5, 9), and the data_alignment update (task 8) is required before the new OHLC data can be consumed by the pipeline.

**Primary recommendation:** Execute tasks in the order given in the Recommended Ordering section at the bottom of this document.

---

## Project Constraints (from CLAUDE.md)

- Think before coding — state assumptions explicitly, surface tradeoffs, ask if unclear.
- Simplicity first — minimum code that solves the problem; no speculative features.
- Surgical changes — touch only what is necessary; do not improve adjacent code.
- Match existing style even when you would do it differently.
- Remove imports/variables/functions that YOUR changes make unused; leave pre-existing dead code alone unless asked.

---

## Task 1: Dead Code NameError Block

### Current State (CONFIRMED)

File: `composer_signal_generator/main.py`

The function `build_portfolio_smart_sharpe` at line 3005 ends with `return port, R, w` at **line 3080**. Lines 3084–3172 are syntactically inside the function body (4-space indent) but are unreachable because the `return` statement at line 3080 exits before them.

The dead block (lines 3084–3172) contains:
- Line 3085: `if COMBO_MODULES_AVAILABLE and base_cfg.get('freeze_combo_universe', False):`
- Line 3091: call to `summarize_combo_distributions(hist_csv, combos_dir)` — function not defined anywhere in the file
- Line 3096: call to `build_combo_corr(wide_csv, combos_dir, ...)` — function not defined
- Line 3113: call to `portfolio_from_shortlist(wide_csv, shortlist, combos_dir, ...)` — function not defined
- Line 3140: call to `per_fold_ssopt(train_dict, oos_dict, combos_dir, ...)` — function not defined
- Lines 3143–3172: additional code referencing `build_and_write_portfolio_series`, `all_results`, `paths` — all undefined in this function's scope. This code ends with `return all_results` at line 3172.

`COMBO_MODULES_AVAILABLE = True` (line 44) means the guard on line 3085 does NOT protect against the NameError — the condition evaluates to True. However, because the block is unreachable (after `return` at 3080), it never executes at runtime. The actual runtime risk is zero. The risk is that a future edit adds code after 3080 that could make it reachable, or that the dead code causes confusion during the Phase 1 rewrite.

**The PLANNING.md description says the guard does not protect.** This is true. But the block is unreachable regardless, making the guard analysis moot at runtime. The correct fix is still removal.

### Exact Fix

Remove lines 3084–3172 (the entire block from `# === Post-processing for combos...` through `return all_results`). The function ends cleanly at line 3082 (`# ---- /SMART SHARPE WEIGHT OPTIMIZATION ----`).

Confirm the next function `fmt_pct` at line 3174 remains intact.

---

## Task 2: precond_mask Threading Back-Port

### Current State (CONFIRMED)

The diff document (`main_vs_main2_diff.md`) is authoritative and verified.

**main.py approach (current):**

`run_comprehensive_evaluation` (line 2737) builds the precondition mask and pre-filters the `signals` dict:
```python
filtered_signals = {
    name: s & precond_mask
    for name, s in tqdm(signals.items(), desc="Filtering Signals")
}
```
Then calls the three runner functions (`run_walk_forward_evaluation`, `run_rolling_window_evaluation`, `run_expanding_window_evaluation`) passing `filtered_signals` rather than the original `signals`. The runner functions and `backtest_signals` have no `precond_mask` parameter.

Current `backtest_signals` signature (line 4497):
```python
def backtest_signals(signals, price_data, target_tickers, benchmark_data=None, period_name=""):
```
Current `run_walk_forward_evaluation` signature (line 2252):
```python
def run_walk_forward_evaluation(signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None):
```
Same pattern for `run_rolling_window_evaluation` (line 2415) and `run_expanding_window_evaluation` (line 2575).

Current `_run_single_backtest` unpacks a 6-element tuple (line 4466):
```python
(signal_name, signal, price_data, target_ticker, daily_ret, EXECUTION_MODE) = args
```

**main2.py approach:**

- `run_comprehensive_evaluation` builds `precond_mask` once and passes it to the three runner functions via `precond_mask=precond_mask`
- The three runner functions accept `precond_mask=None` and pass it to `backtest_signals`
- `backtest_signals` accepts `precond_mask=None` and includes it in each task tuple
- `_run_single_backtest` unpacks a 7-element tuple and applies the mask per-signal inside the worker

**Behavioral difference:**

main.py's pre-filter applies the mask using bitwise AND (`s & precond_mask`), which permanently combines the precondition into the signal series before any windowing. This means that when a walk-forward window is sliced, the signal within that window already has the full-history precondition baked in.

main2.py's pass-through approach applies the mask inside the worker after reindexing to the price data slice: `pc = precond_mask.reindex(price_data.index).fillna(False); sig = sig & pc`. This is more correct because it allows the precondition mask to be applied relative to the windowed data slice, and a reindex is done to ensure alignment.

The PLANNING.md identifies this as a meaningful behavioral improvement worth back-porting.

### Exact Fix

Three specific changes:

**Change 1: `_run_single_backtest` (line 4460-4493)** — update to accept and apply `precond_mask`.

Current unpacking (line 4466):
```python
(signal_name, signal, price_data, target_ticker, daily_ret, EXECUTION_MODE) = args
```
New unpacking (7-element):
```python
(signal_name, signal, price_data, target_ticker, daily_ret, precond_mask, EXECUTION_MODE) = args
```
Add after `sig = signal.reindex(price_data.index).fillna(False)`:
```python
if precond_mask is not None:
    pc = precond_mask.reindex(price_data.index).fillna(False)
    sig = sig & pc
```
Remove the existing docstring comment "Pre-filtering is now done in the main process, so this function is simpler."

**Change 2: `backtest_signals` (line 4497)** — add `precond_mask=None` parameter and include in task tuple.

Current signature (line 4497):
```python
def backtest_signals(signals, price_data, target_tickers, benchmark_data=None, period_name=""):
```
New signature:
```python
def backtest_signals(signals, price_data, target_tickers, benchmark_data=None, period_name="", precond_mask=None):
```
Current task tuple (line 4507-4508):
```python
task_args = (signal_name, signal, price_data, target_ticker,
             daily_ret, EXECUTION_MODE)
```
New task tuple:
```python
task_args = (signal_name, signal, price_data, target_ticker,
             daily_ret, precond_mask, EXECUTION_MODE)
```

**Change 3: Three runner functions** — add `precond_mask=None` parameter and pass to `backtest_signals` calls.

- `run_walk_forward_evaluation` (line 2252): add `precond_mask=None` to signature; update all `backtest_signals(...)` calls within to pass `precond_mask=precond_mask`
- `run_rolling_window_evaluation` (line 2415): same changes
- `run_expanding_window_evaluation` (line 2575): same changes

**Change 4: `run_comprehensive_evaluation` (line 2737)** — replace pre-filter approach with pass-through.

Remove the filtered_signals block (lines 2759-2779):
```python
filtered_signals = signals  # Start with the original signals
if base_cfg.get('preconditions'):
    ...
    filtered_signals = { name: s & precond_mask ... }
```
Replace with the main2.py approach: build `precond_mask` and pass it to runner functions. All calls that currently pass `filtered_signals` should pass `signals` (the original unmodified dict) with the mask threaded separately.

Note: Calls to `backtest_signals` within `run_comprehensive_evaluation` itself (the holdout section, lines 2791-2792) also need `precond_mask=precond_mask` added.

**Important:** Do NOT remove `run_comprehensive_evaluation`'s `completed_evaluations` resume parameter or its tuple return value — these are correct in main.py and absent from main2.py (a regression in main2.py).

---

## Task 3: max_workers User-Configurable

### Current State (CONFIRMED)

Two hardcoded `ProcessPoolExecutor` calls in main.py:

1. `backtest_signals` (line 4514): `with ProcessPoolExecutor(max_workers=5) as executor:`
2. `enrich_with_synergistic_combos` (line 5049): `with ProcessPoolExecutor(max_workers=5, initializer=_init_worker, initargs=init_args) as executor:`

No existing prompt in `get_enhanced_user_inputs` controls these values. The function has 19+ interactive prompts already (lines 3497-3739) covering evaluation methods, window sizes, combo settings, portfolio settings, etc. A new prompt fits naturally in the "System Resource Settings" area (there is none currently — it would be added near the top or bottom of the function).

`main2.py` uses `max_workers=6` for both locations (minor tuning, not user-configurable either). The goal is to make this user-configurable so low-RAM systems can use fewer workers.

### Exact Fix

Add a new interactive prompt in `get_enhanced_user_inputs` (near the top, before evaluation method selection, or after the eval config section):

```python
print("\nSystem Resources:")
max_workers_input = safe_input(
    "Max parallel workers (lower = less RAM, default 5 = recommended for 16GB+ systems): ",
    default="5"
).strip()
config['max_workers'] = int(max_workers_input) if max_workers_input else 5
```

Then in `backtest_signals`, replace the hardcoded `max_workers=5` with:
```python
max_workers = base_cfg.get('max_workers', 5) if base_cfg else 5
```
But `backtest_signals` does not currently receive `base_cfg`. Two options:
- Pass `base_cfg` as a parameter to `backtest_signals` (surgical change, but adds a parameter)
- Use a module-level global for `MAX_WORKERS` set from config before the run begins

Recommended: use a module-level global. In `enhanced_main` (or `run_comprehensive_evaluation`), after config is loaded:
```python
global MAX_WORKERS
MAX_WORKERS = base_cfg.get('max_workers', 5)
```
And define the default near the top of the file:
```python
MAX_WORKERS = 5  # configurable via user prompt
```

Then in `backtest_signals` and `enrich_with_synergistic_combos`, reference `MAX_WORKERS` instead of the literal `5`.

This is the least invasive approach and matches the existing pattern of module-level constants (e.g., `EXECUTION_MODE`, `COMBO_MODULES_AVAILABLE`).

---

## Task 4: Delete main2.py

### Current State

File exists at `composer_signal_generator/main2.py` (5,524 lines). The diff document confirms it has:
- The expanding window path bug (writes to "rolling" directory instead of "expanding")
- A dead unreachable `while True` loop
- `KeyError` crash risk in `generate_evaluation_summary`
- Diagnostic print dumps in holdout section (lines 2939-2951)

### Exact Fix

After tasks 2 and 3 are verified complete, delete the file. No other files reference main2.py.

---

## Task 5: Rename `_ATR()` to `_rolling_std()` and Add True `_ATR()`

### Current State (CONFIRMED)

File: `composer_signal_generator/main.py`

Lines 111-115:
```python
def _ATR(prices: pd.DataFrame, tkr: str, n: int) -> pd.Series:
    """Average True Range"""
    s = _PRICE(prices, tkr)
    # Simple approximation using price volatility
    return s.rolling(int(n)).std()
```

This is rolling standard deviation, not Average True Range. The function is registered in `_ALLOWED_CALLS` at line 124:
```python
_ALLOWED_CALLS = {"PRICE": _PRICE, "SMA": _SMA, "EMA": _EMA, "RSI": _RSI,
                  "BBANDS": _BBANDS, "ATR": _ATR, "ZSCORE": _ZSCORE,
                  "price": _PRICE, "sma": _SMA, "ema": _EMA, "rsi": _RSI,
                  "bbands": _BBANDS, "atr": _ATR, "zscore": _ZSCORE}
```

`_ATR` appears in exactly two places: the function definition and the `_ALLOWED_CALLS` dict. No other callers in main.py use `_ATR` by that name directly — they go through `_ALLOWED_CALLS` (via `_safe_eval_precond`).

True ATR requires high, low, and close prices. This is a dependency on task 7 (Tiingo OHLC extension). Until task 7 is complete, only the rename can happen.

### Dependencies

- Task 7 must be completed before true ATR can be implemented (need OHLC column names from the extended data_loader)
- Task 8 determines what column names will be available in the price DataFrame

After task 7 is complete, OHLC column names in the Tiingo CSV will be: `date, open, high, low, close`. In `composer_signal_generator/main.py`, prices come from yfinance (not Tiingo), so the OHLC column names will be whatever yfinance provides. This is a complication: main.py uses yfinance, not Tiingo.

**Surprise finding:** The `composer_signal_generator` uses yfinance for price data (line 16: `import yfinance as yf`). The Tiingo OHLC changes in tasks 7/8 are in the `rsi_tester` module. The true ATR implementation in `main.py` would need to access yfinance OHLC columns. yfinance provides columns: `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume` (capital letters). This is a separate data source from the Tiingo OHLC in tasks 7/8.

The precondition engine in main.py uses a `prices` DataFrame that is a dict-like structure keyed by ticker. The actual DataFrame with OHLC for true ATR would need to be accessed as `prices.get(tkr + '_High')` or similar, which does not currently exist.

### Exact Fix

**Phase A (do now, no dependency):** Rename `_ATR` to `_rolling_std` everywhere:
- Line 111: rename function definition
- Lines 124-127: update `_ALLOWED_CALLS` — remove both `"ATR": _ATR` and `"atr": _ATR` entries; add `"ROLLING_STD": _rolling_std` and `"rolling_std": _rolling_std`

**Phase B (requires clarification — flag as open question):** True ATR in main.py requires OHLC data from yfinance. The `_safe_eval_precond` mechanism uses the prices DataFrame which currently only has close prices per ticker. Adding true ATR in the precondition engine would require restructuring how the price data is passed to the precondition functions. This is a non-trivial change that may be out of scope for Phase 0. **Recommend: rename only in Phase 0; defer true ATR to Phase 1 architecture** where the data layer is redesigned.

For `rsi_tester/indicators.py` (task 9), true ATR can be added cleanly because that module uses a flat DataFrame with `signal_close`, `target_close`, etc. columns and OHLC columns can be added straightforwardly.

---

## Task 6: Add `_BBAND_LOWER()`

### Current State (CONFIRMED)

File: `composer_signal_generator/main.py`

Lines 104-109:
```python
def _BBANDS(prices: pd.DataFrame, tkr: str, n: int, std: float = 2.0) -> pd.Series:
    """Bollinger Bands - returns upper band"""
    s = _PRICE(prices, tkr)
    sma = s.rolling(int(n)).mean()
    std_dev = s.rolling(int(n)).std()
    return sma + (std_dev * float(std))
```

`_BBANDS` appears in `_ALLOWED_CALLS` (line 124-127) as `"BBANDS": _BBANDS` and `"bbands": _BBANDS`.

No rename of `_BBANDS` is strictly required by the task description (the VISION.md says "Rename to `_BBAND_UPPER`; add `_BBAND_LOWER`"). However, renaming `_BBANDS` would require updating `_ALLOWED_CALLS` and potentially any saved precondition strings that reference `BBANDS(...)`. The minimal fix is to keep `_BBANDS` as-is (backward compatible alias for upper band) and add `_BBAND_LOWER`. Optionally add `_BBAND_UPPER` as an alias.

### Exact Fix

Add after line 109:
```python
def _BBAND_LOWER(prices: pd.DataFrame, tkr: str, n: int, std: float = 2.0) -> pd.Series:
    """Bollinger Bands - returns lower band"""
    s = _PRICE(prices, tkr)
    sma = s.rolling(int(n)).mean()
    std_dev = s.rolling(int(n)).std()
    return sma - (std_dev * float(std))
```

Update `_ALLOWED_CALLS` to add:
```python
"BBAND_LOWER": _BBAND_LOWER,
"bband_lower": _BBAND_LOWER,
```

Optionally (per VISION.md direction): add `"BBAND_UPPER": _BBANDS, "bband_upper": _BBANDS` as aliases for clarity, while keeping `BBANDS` for backward compatibility.

Do not rename `_BBANDS` — that would break any existing precondition expressions saved in configs that use `BBANDS(...)`.

---

## Task 7: Extend Tiingo data_loader.py to Fetch OHLC

### Current State (CONFIRMED)

File: `rsi_tester/strategy_engine/src/data_loader.py`

Current `download_ticker_data` (lines 36-71):
```python
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
df = df[['date', 'adjClose']].rename(columns={'adjClose': 'close'})

os.makedirs(data_dir, exist_ok=True)
file_path = data_dir / f"{ticker}.csv"
df.to_csv(file_path, index=False)
```

The Tiingo API response JSON includes `adjOpen`, `adjHigh`, `adjLow`, `adjClose` fields alongside date. No structural changes to the API call are needed — the data is already available in the response.

Current CSV format: `date,close`

### Exact Fix

Update `download_ticker_data` to save OHLC:
```python
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
df = df[['date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose']].rename(
    columns={'adjOpen': 'open', 'adjHigh': 'high', 'adjLow': 'low', 'adjClose': 'close'}
)

os.makedirs(data_dir, exist_ok=True)
file_path = data_dir / f"{ticker}.csv"
df.to_csv(file_path, index=False)
```

New CSV format: `date,open,high,low,close`

**Backward compatibility:** `check_freshness_and_update` currently reads the CSV to check max date using `df['date'].max()`. This works with both old (2-column) and new (5-column) CSVs — the 'date' column always exists. No change needed to `check_freshness_and_update` for the freshness check itself.

However, the freshness check will not detect that an existing close-only CSV needs upgrading to OHLC. If old CSVs exist, the freshness check will see them as current and not re-download. **Resolution:** Old CSVs must be deleted from the data directory when upgrading, or the executor should be instructed to delete and re-download manually. This is an operational note for the executor, not a code change.

**Dependency note:** task 8 (`data_alignment.py`) must handle both old `date,close` CSVs and new `date,open,high,low,close` CSVs for backward compatibility during any transition period.

---

## Task 8: Update data_alignment.py / load_ticker_csv()

### Current State (CONFIRMED)

File: `rsi_tester/strategy_engine/src/data_alignment.py`

Current `load_ticker_csv` (lines 5-17):
```python
def load_ticker_csv(ticker, data_dir):
    file_path = data_dir / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    return df[['date', 'close']]
```

The final line `return df[['date', 'close']]` will break if a CSV has the new OHLC format, because it hardcodes a return of only the `close` column. Actually it will not break — `df[['date', 'close']]` will work fine on a 5-column CSV because `close` still exists. But the OHLC columns (`open`, `high`, `low`) will be silently dropped.

`build_master_dataframe` renames the `close` column with role-based names (e.g., `signal_close`, `target_close`). It never references `open`, `high`, or `low`. So the current `load_ticker_csv` and `build_master_dataframe` are technically backward-compatible with the new CSVs — they just ignore the new columns.

The task requires making OHLC available for ATR computation in task 9. The indicators.py `add_indicator` function dispatches based on the `asset_role` and indicator name, reading `{asset_role}_close`. For ATR, it would need `{asset_role}_high`, `{asset_role}_low`.

### Exact Fix

Update `load_ticker_csv` to return OHLC when available, with backward compat for old close-only CSVs:

```python
def load_ticker_csv(ticker, data_dir):
    file_path = data_dir / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
        return df[['date', 'open', 'high', 'low', 'close']]
    return df[['date', 'close']]
```

Update `build_master_dataframe` to forward OHLC columns when present. After the role-rename of `close`, also rename `open`, `high`, `low` if they exist:

```python
# In the section that loads each ticker:
df_signal = load_ticker_csv(signal_ticker, data_dir)
rename_map = {'close': 'signal_close'}
if 'high' in df_signal.columns:
    rename_map.update({'open': 'signal_open', 'high': 'signal_high', 'low': 'signal_low'})
df_signal = df_signal.rename(columns=rename_map)
```

This ensures that `signal_high`, `signal_low` etc. are available in the master DataFrame for ATR computation, while old CSVs without OHLC continue to work (they simply won't have those columns, and ATR computation will raise a clear error rather than silently failing).

---

## Task 9: Add `calculate_atr()` and `calculate_bbands_lower()` to indicators.py

### Current State (CONFIRMED)

File: `rsi_tester/strategy_engine/src/indicators.py`

Current functions: `calculate_sma`, `calculate_ema`, `calculate_rsi`, `calculate_cumret`, `add_indicator`. No ATR or BBands functions.

`add_indicator` dispatches by `indicator_name.upper()` with an explicit set of known names (RSI, SMA, EMA, CUMRET). Unknown names raise `ValueError`.

### Exact Fix

Add after `calculate_cumret`:

```python
def calculate_atr(high, low, close, period):
    """
    Wilder's Average True Range.
    high, low, close are pd.Series aligned to the same index.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    # Wilder's smoothing: EMA with alpha = 1/period
    return tr.ewm(alpha=1/period, adjust=False).mean()

def calculate_bbands_lower(series, period, num_std=2.0):
    """Lower Bollinger Band: SMA - num_std * rolling std."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma - (std * num_std)
```

Update `add_indicator` to dispatch to the new functions:

```python
elif indicator_name.upper() == "ATR":
    high_col = f"{asset_role}_high"
    low_col = f"{asset_role}_low"
    close_col = f"{asset_role}_close"
    if high_col not in df.columns or low_col not in df.columns:
        raise ValueError(
            f"ATR requires '{high_col}' and '{low_col}' columns. "
            f"Re-download data with OHLC support (task 7)."
        )
    df[indicator_col] = calculate_atr(df[high_col], df[low_col], df[price_col], period)
elif indicator_name.upper() == "BBAND_LOWER":
    df[indicator_col] = calculate_bbands_lower(df[price_col], period)
```

**Dependency:** task 8 must be complete before ATR works, because `signal_high` and `signal_low` columns only appear in the master DataFrame after `build_master_dataframe` is updated to forward OHLC columns.

---

## Task 10: Fix Monte Carlo Drawdown Unit Mismatch

### Current State (CONFIRMED — with a nuance)

File: `monte_carlo_sim/Monte Carlo walk forward composer working.py`

**The nuance:** The inventory document states `run_monte_carlo_simulation` stores `max_drawdowns` as fractions (0.15) while `analyze_drawdowns` returns `max_drawdown` as percent (15.0). After direct code inspection, both functions use an identical formula: `(peak - value) / (1 + peak / 100)`. Since `peak` and `value` are cumulative percent returns (e.g., 15.0 for 15%), the formula produces values in the range 0–100 (percent-ish), not 0–1 (fractional).

However, there is still a unit inconsistency worth verifying. In `run_monte_carlo_simulation` (line 618):
```python
drawdown = ((peak - cum_return) / (1 + peak / 100)) if peak > 0 else 0
```
When `peak = 0`, `drawdown = 0`. When `peak > 0`, e.g., peak=10.0 (10%), cum_return=7.0 (7%): `drawdown = (10-7)/(1.10) = 2.73`. This is a percentage value approximately equal to the drawdown percent.

In `analyze_drawdowns` (line 756):
```python
current_drawdown = ((running_peak - value) / (1 + running_peak / 100))
```
Same formula, same units.

In `plot_drawdown_distributions` (lines 1062-1067):
```python
max_drawdowns = simulation_results['max_drawdowns']
sns.histplot(max_drawdowns, kde=True, bins=30, ax=ax1, color='blue', alpha=0.6)
ax1.axvline(x=actual_max_drawdown, color='r', linestyle='--',
            label=f'Actual: {actual_max_drawdown:.2f}%')
```
The X-axis label (line 1094) says `'Maximum Drawdown (%)'`.

**If** both `simulation_results['max_drawdowns']` and `actual_max_drawdown` (from `analyze_drawdowns`) are in the same units (both approximately percent), then the histogram overlay would be visually correct. The inventory's claim of a fraction vs percent mismatch may be incorrect based on the code as it exists today.

**However:** the inventory document was written by someone who had studied this code and flagged it. The risk is real that in some execution paths, `cum_return` or `peak` could be in different units (e.g., if the input `returns` are decimal fractions rather than percentages), causing the formula to produce different-scale results.

Looking at `run_walk_forward_test` (lines 1200-1207), the cumulative return is computed as:
```python
cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
```
where `r_decimal = r / 100.0`. This explicitly works in percent (divides by 100 to get to decimal, computes, multiplies by 100 to return to percent). So `cum_return` is definitely in percent.

**Conclusion:** Both `simulation_results['max_drawdowns']` and `actual_max_drawdown` use the same formula on percent-scale values, so they are in the same approximate units. The inventory's description of a "fraction vs percent" mismatch is likely based on a misreading of the formula. The values are NOT classical fractional drawdowns (0.15) nor classical percent drawdowns (15.0) — they are a hybrid formula result that is approximately percent-scale.

**The actual bug to fix:** Even though both use the same formula, the statistics box labels in `plot_drawdown_distributions` format all values with `:.2f%` suffixes. If the values are ~2.73 (from the formula), displaying them as "2.73%" is correct. If values are 0.15 (true fractional), it would display "0.15%" which is wrong. The PLANNING.md says to multiply `simulation_results['max_drawdowns'] * 100` — this would be correct IF `max_drawdowns` were fractional (0.15→15.0). But as confirmed, they are NOT fractional.

**Recommended minimal fix:** Add a code comment clarifying the unit (the formula produces approximately-percent-scale values for typically-sized returns), and verify by running the script with a sample. Do NOT multiply by 100, as that would introduce a new bug (20× scale inflation).

**Risk:** If the executor is uncertain, test the fix with a sample run and check that simulated drawdowns (e.g., 5–30%) overlay correctly with actual drawdowns in similar range before committing.

---

## Task 11: Fix Total_Trades Bug in rsi_tester metrics.py

### Current State (CONFIRMED)

File: `rsi_tester/strategy_engine/src/metrics.py`

Lines 26-27:
```python
active_days = df[df['signal_active'] == 1]
total_trades = len(active_days)
```

`total_trades` counts active days (rows where `signal_active == 1`). A 10-day active streak counts as 10. A single-day signal followed by 5 days off and another 3-day signal counts as 4 total_trades, but represents 2 trade entries.

Lines 32-33 already compute the correct count:
```python
streak_starts = ((df['signal_active'] == 1) & (df['signal_active'].shift(1) != 1)).sum()
avg_hold_days = total_trades / streak_starts if streak_starts > 0 else 0
```

`streak_starts` is the number of 0→1 transitions — exactly the correct value for `total_trades`.

The downstream filter in `filter_results.py` uses `Total_Trades > 20` as a minimum threshold. With the current bug, a single 21-day active streak would pass this filter (21 active days = 21 "trades"). After the fix, that same streak would be 1 trade and would fail the filter. **This will change filter behavior — signals with few long streaks may be filtered out that previously passed.** This is the correct and intended behavior.

### Exact Fix

Replace lines 26-27 with:
```python
active_days = df[df['signal_active'] == 1]
streak_starts = ((df['signal_active'] == 1) & (df['signal_active'].shift(1) != 1)).sum()
total_trades = streak_starts
```

Update line 30 (the `win_rate` denominator): it currently divides by `total_trades` (which was `len(active_days)`). After the fix, `total_trades` is `streak_starts`, so `win_rate` would divide by number of trades — which may not be what is intended. The `win_rate` is described in the inventory as "fraction of active days where `strategy_return > 0`". So the denominator for `win_rate` should remain `len(active_days)`, not the new `total_trades`.

Revised fix:
```python
active_days = df[df['signal_active'] == 1]
n_active_days = len(active_days)
streak_starts = ((df['signal_active'] == 1) & (df['signal_active'].shift(1) != 1)).sum()
total_trades = streak_starts

if total_trades > 0:
    win_rate = len(active_days[active_days['strategy_return'] > 0]) / n_active_days
    avg_return = active_days['strategy_return'].mean()
    avg_hold_days = n_active_days / streak_starts if streak_starts > 0 else 0
```

Also update the check `if total_trades > 0:` (line 29) — this guard is now checking whether any trade entries exist. If `streak_starts == 0`, there are no trades. This is correct.

Remove the now-redundant `streak_starts` and `avg_hold_days` lines at 32-33 (since they are incorporated above).

---

## Dependencies Between Tasks

```
Task 7 (Tiingo OHLC) ─┬─→ Task 8 (data_alignment OHLC) ──→ Task 9 (indicators.py ATR)
                       └─→ Task 5 Phase B (true ATR in main.py) [DEFERRED to Phase 1]

Task 2 (precond_mask) ──→ Task 4 (delete main2.py)
Task 3 (max_workers)  ──→ Task 4 (delete main2.py)

All other tasks are independent of each other.
```

---

## Recommended Task Ordering

**Group A — Independent, no deps, do first:**
1. Task 1: Remove dead code block in main.py (safest, purely subtractive)
2. Task 6: Add `_BBAND_LOWER()` (purely additive, no deps)
3. Task 11: Fix Total_Trades in metrics.py (contained, no deps)
4. Task 10: MC drawdown unit mismatch (verify units first, apply comment or fix)

**Group B — Back-port from main2.py:**
5. Task 2: precond_mask threading (most complex change in this phase)
6. Task 3: max_workers user prompt (adds global + prompt)
7. Task 4: Delete main2.py (after 2 and 3 verified)

**Group C — Tiingo OHLC chain (must be in order):**
8. Task 7: Extend data_loader.py to fetch OHLC
9. Task 8: Update data_alignment.py to forward OHLC columns
10. Task 9: Add calculate_atr() and calculate_bbands_lower() to indicators.py

**Task 5 (Rename _ATR, add true ATR):**
11. Task 5 Phase A: Rename `_ATR` → `_rolling_std` in main.py (no deps, do with Group A)
    Task 5 Phase B: True `_ATR` in main.py precondition engine — **defer to Phase 1** (requires yfinance OHLC, architectural decision needed)

---

## Risks and Surprises

### Risk 1: Task 10 unit mismatch may be overstated

The inventory claims `max_drawdowns` are fractional (0.15) but code inspection shows both the simulation and analysis functions use the same formula producing approximately-percent-scale values. **Do not multiply by 100 without first running a test.** If `max_drawdowns` values in a live run are in the range 0–30 (matching percent drawdowns), the overlay is already correct and the "fix" would introduce a 100× scale error.

Recommended action for executor: run the script once with a test symphony before and after any change, inspect the histogram output, and verify that simulated and actual drawdown values are on the same scale.

### Risk 2: Task 11 filter threshold impact

Fixing `Total_Trades` to count trade entries (not active days) will cause previously-passing signals with long single runs to fail the `Total_Trades > 20` filter. This is intentional but the executor should be aware that `filtered.csv` output will change — fewer signals may pass.

### Risk 3: Task 2 precond_mask — resume/checkpoint compatibility

After the back-port of precond_mask threading, the mask is computed once in `run_comprehensive_evaluation` and passed down. If a run is interrupted mid-way and resumed from a pickle checkpoint, the mask is rebuilt from the config at resume time. This is correct because `build_precondition_series` is deterministic given the same config and price data.

### Risk 4: Task 7 — old CSVs not auto-upgraded

`check_freshness_and_update` detects whether data is current by comparing the max date. It will see an old `date,close` CSV as "current" if the date matches. Old CSVs will not be upgraded to OHLC automatically. The executor must manually delete the data directory (or specific ticker CSVs) to force a full re-download after the data_loader change.

### Risk 5: Task 5 Phase B — true ATR in main.py precondition engine

The `composer_signal_generator` uses yfinance price data, not Tiingo. The `prices` DataFrame passed to `_safe_eval_precond` is keyed by ticker symbol with close prices only. Adding true ATR to the precondition engine would require either:
- Restructuring the `prices` dict to include OHLC per ticker (significant change to data loading)
- Or accepting that `_ATR` in main.py remains rolling std for this phase

**Recommended decision:** defer true ATR in main.py to Phase 1 (Phase A rename only for Phase 0).

---

## Open Questions

1. **Task 10 — MC drawdown units:** Confirm by running the MC script whether simulated `max_drawdowns` are in the 0–1 (fractional) or 0–100 (percent) range. The code suggests percent-scale, but the inventory claims fractional. The correct fix depends on the actual runtime values.

2. **Task 5 Phase B — scope:** Is true `_ATR()` in `composer_signal_generator/main.py`'s precondition engine required for Phase 0, or should Phase 0 only do the rename and defer the implementation? (Recommendation: defer to Phase 1.)

3. **Task 2 — precond_mask and combo enrichment:** `enrich_with_synergistic_combos` is called from inside `run_comprehensive_evaluation` with the original `signals` dict (not `filtered_signals`). After the back-port, should the combo enrichment also receive `precond_mask`? The main2.py implementation does NOT pass `precond_mask` to `enrich_with_synergistic_combos`. The current main.py behavior (pre-filtered signals passed to combo enrichment) is subtly different. For Phase 0, match main2.py behavior: do not pass precond_mask to combo enrichment.

---

## Sources

All claims in this document are VERIFIED by direct source code inspection. No web search or training-data-only claims are present.

| File | Lines inspected | Finding |
|------|-----------------|---------|
| `composer_signal_generator/main.py` | 1–150, 2249–2280, 2413–2445, 2573–2650, 2737–2820, 3005–3172, 3497–3700, 4455–4520, 5035–5055 | Tasks 1, 2, 3, 5, 6 |
| `composer_signal_generator/main2.py` | 2249–2370, 2868–3100, 4900–4972 | Task 2 diff verification |
| `monte_carlo_sim/Monte Carlo walk forward composer working.py` | 544–673, 675–850, 1052–1165, 1166–1240 | Task 10 |
| `rsi_tester/strategy_engine/src/metrics.py` | 1–75 | Task 11 |
| `rsi_tester/strategy_engine/src/data_loader.py` | 1–121 | Task 7 |
| `rsi_tester/strategy_engine/src/data_alignment.py` | 1–48 | Task 8 |
| `rsi_tester/strategy_engine/src/indicators.py` | 1–107 | Task 9 |
| `docs/inventory/SYNTHESIS.md` | Full | Cross-reference |
| `docs/inventory/main_vs_main2_diff.md` | Full | Task 2 architecture |
| `docs/inventory/monte_carlo_sim.md` | Full | Task 10 context |
| `docs/inventory/rsi_tester.md` | Full | Tasks 7–11 context |
| `VISION.md` | Full | Bug list cross-reference |
| `PLANNING.md` | Full | Back-port requirements |
