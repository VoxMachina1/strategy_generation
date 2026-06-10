# Exhaustive Diff: main.py vs main2.py

**Date produced:** 2026-06-08  
**Files compared:**
- FILE A: `composer_signal_generator/main.py`
- FILE B: `composer_signal_generator/main2.py`

---

## 1. Exact Line Counts

| File | Lines |
|------|-------|
| main.py | 5,070 |
| main2.py | 5,524 |
| Difference | +454 lines in main2.py |

---

## 2. Sections Unique to main2.py (present in B, absent or different in A)

### 2a. `enhanced_main()` — Resume/Checkpoint Architecture REMOVED (main2.py lines 4240–4554)

main2.py's `enhanced_main()` is a simpler, single-pass execution loop. It does **not** have the resume-from-checkpoint system that main.py has. It calls `run_comprehensive_evaluation()` once per loop iteration, collects `all_results` as a plain value (not a tuple), and ends with a `safe_input("Run another evaluation...? [y/N]")` prompt that loops.

Unique elements in main2.py's `enhanced_main`:
- Inline blackout adjustment logic with `adjusted_blackout_ranges` (main.py delegates to `apply_blackout_ranges` directly)
- A call to `normalize_price_panel(full_price_data)` before signal generation (main.py does not call this)
- A run manifest/README write at the end of each pass: `write_manifest(paths)` and `write_readme(paths, cfg_summary)` with a formatted `cfg_summary` string (main.py has no equivalent)
- A **duplicated dead `while True` loop** at lines 4477–4499 that is unreachable (the first loop's `break` never reaches it); this appears to be a copy-paste artifact and is a latent bug

### 2b. `backtest_signals()` — `precond_mask` Parameter (main2.py line 4947)

```python
# main2.py
def backtest_signals(signals, price_data, target_tickers, benchmark_data=None, period_name="", precond_mask=None):
```
main2.py's `backtest_signals` accepts and passes `precond_mask` into each worker task tuple. The parallel worker `_run_single_backtest` in main2.py applies the mask inline:
```python
if precond_mask is not None:
    pc = precond_mask.reindex(price_data.index).fillna(False)
    sig = sig & pc
```

### 2c. `ProcessPoolExecutor` in `backtest_signals` — `max_workers=6` (main2.py line 4966)

main2.py uses `max_workers=6` inside `backtest_signals`. main.py uses `max_workers=5`.

### 2d. `ProcessPoolExecutor` in `enrich_with_synergistic_combos` — `max_workers=6` (main2.py line 5503)

main2.py uses `max_workers=6`. main.py uses `max_workers=5`.

---

## 3. Sections Unique to main.py (present in A, absent or different in B)

### 3a. `BASE_OUTPUT_DIR` Constant (main.py line 14)

```python
BASE_OUTPUT_DIR = Path(__file__).parent / "datasets"
```
This constant is used by main.py's `enhanced_main()` to construct the parent output directory. It is absent from main2.py, which instead computes `outdir` directly from `run_name`.

### 3b. `enhanced_main()` — Full Resume/Checkpoint System (main.py lines 3960–4113)

main.py's `enhanced_main()` is substantially longer and more robust. Key elements absent from main2.py:

- **Prompts for a run name first**, then checks for a `master_checkpoint.pkl` file in `BASE_OUTPUT_DIR / run_name`
- **Loads full state from checkpoint** (signals, price_data, base_cfg, blackout_ranges, completed_evaluations) if found
- **Stateful evaluation loop**: iterates through `original_plan` (list of EvalModes from config), finds the next uncompleted mode, runs exactly one mode per outer loop iteration
- **Updates master checkpoint after each mode completes**, saving `completed_evaluations` so interrupted runs resume from the correct position
- **Deletes the master checkpoint** on clean completion
- **Returns a tuple** from `run_comprehensive_evaluation`: `(all_results, completed_evaluations)`
- Passes `completed_evaluations=completed_evaluations` to `run_comprehensive_evaluation`

### 3c. `run_comprehensive_evaluation()` — `completed_evaluations` Resume Parameter (main.py line 2737)

```python
# main.py
def run_comprehensive_evaluation(signals, price_data, target_tickers, eval_modes, config, paths,
                                  name_prefix="", base_cfg=None, completed_evaluations=None):
```
- Accepts `completed_evaluations` list for resume-aware skipping
- Returns `(all_results, completed_evaluations)` tuple
- **Pre-filters signals dict** using the precondition mask before calling any runner:
  ```python
  filtered_signals = {k: v for k, v in signals.items() if precond_mask is None or (v & precond_mask).any()}
  ```
  Then passes `filtered_signals` (not `precond_mask`) to the three runner functions

### 3d. `generate_evaluation_summary()` — `top_n_per_method=50`, safe dict access (main.py line 3333)

```python
# main.py
def generate_evaluation_summary(..., top_n_per_method=50):
    ...
    holdout_data = results.get('holdout', {})
    filtered = holdout_data.get('filtered_results', pd.DataFrame())
```
- Default `top_n_per_method=50`
- Uses `.get()` for safe dict access — will not crash if 'filtered_results' key is missing
- Outputs results only to CSV (no verbose console printing)

### 3e. `_run_single_backtest()` — No `precond_mask` in arg tuple (main.py line 4460)

main.py's worker function unpacks a **6-element tuple**:
```python
(signal_name, signal, price_data, target_ticker, daily_ret, EXECUTION_MODE) = args
```
The signal passed in is already precondition-filtered by the caller, so no mask is needed inside the worker. The docstring explicitly notes: "Pre-filtering is now done in the main process, so this function is simpler."

---

## 4. Sections Confirmed Identical

The following sections are byte-for-byte identical between both files (same logic, same line structure):

- Lines 1–13: All module-level imports
- Lines 15–351: All constants, class definitions (`EvalMode`, `EvaluationConfig`, `RunPaths`, helper functions through `download_prices_max_debug`)
- Lines 353–2250: All of the following: `get_preconditions_from_user`, `build_precondition_mask`, `normalize_price_panel`, `apply_blackout_ranges`, `safe_input`, `safe_print`, `_cleanup_tqdm`, `_tqdm`, `PM` class, `_parse_periods`, `unique`, `_series`, `_as_results_dict`, `_is_combo_row`, `export_all_combo_artifacts`, `build_portfolio_smart_sharpe`, `write_manifest`, `write_readme`, `generate_evaluation_summary` (function body — only default parameter differs), `backtest_signals` (function signature differs — see section 2b)
- `_sanitize_returns`, `calculate_robustness_score`, `apply_comprehensive_filter`, `apply_robustness_filter`, `generate_composer_output`, `save_composer_signals`, `display_composer_preview`, `report_data_availability`, `get_initial_price_data`, `calculate_quantstats_metrics` — identical in both files
- `merge_train_test_results`, `get_user_inputs`, `_combine_series`, `_combine_many`, `_get_metric`, `_greedy_build_combo`, `_init_worker`, `_find_combos_for_primary`, `enrich_with_synergistic_combos` — identical except for the `max_workers` constant in `enrich_with_synergistic_combos` (5 vs 6)
- `generate_signals` — identical in both files
- `get_enhanced_user_inputs`, `get_eval_only` — identical in both files
- `convert_signal_to_composer_format` — identical in both files
- `if __name__ == "__main__":` block — identical in both files

---

## 5. Value Differences (Constants / Parameters)

| Location | Parameter | main.py value | main2.py value |
|----------|-----------|---------------|----------------|
| ~line 352 | `ThreadPoolExecutor(max_workers=...)` in `download_prices_max_debug` | `10` | `12` |
| `generate_evaluation_summary` default | `top_n_per_method` | `50` | `10` |
| `backtest_signals` | `ProcessPoolExecutor(max_workers=...)` | `5` | `6` |
| `enrich_with_synergistic_combos` | `ProcessPoolExecutor(max_workers=...)` | `5` | `6` |

---

## 6. Functional Differences

### Difference F1 — Precondition Architecture (MAJOR)

The two files implement precondition filtering using opposite strategies:

**main.py approach (pre-filter):** `run_comprehensive_evaluation` builds the `precond_mask` once, then filters the `signals` dict down to only signals that fire at least once under the mask. The filtered dict is passed to the runner functions. The runners and `backtest_signals` are unaware of preconditions.

**main2.py approach (pass-through mask):** `run_comprehensive_evaluation` builds `precond_mask` once, then passes it explicitly as a parameter to all three runner functions (`run_walk_forward_evaluation`, `run_rolling_window_evaluation`, `run_expanding_window_evaluation`), which in turn pass it to `backtest_signals`, which passes it into each worker task tuple. The mask is applied per-signal inside `_run_single_backtest`.

**Behavioral consequence:** The two approaches produce different results when a signal partially overlaps with a precondition mask. The pre-filter approach (main.py) drops signals that never fire under the mask. The pass-through approach (main2.py) keeps all signals in the dict and applies the mask at backtest time, which is more correct for time-windowed evaluations because the mask can vary across windows.

### Difference F2 — `run_walk_forward_evaluation`, `run_rolling_window_evaluation`, `run_expanding_window_evaluation` Signatures

main2.py adds `precond_mask=None` to all three runner function signatures (main.py ~lines 2252, 2415, 2575 vs main2.py ~lines 2251, 2472, 2686). The function bodies also differ in:
- main2.py wraps iteration `range()` calls with `_tqdm()` for progress bars; main.py uses plain `range()`
- main.py has "DIAGNOSTIC H3" debug print statements in checkpoint messages; main2.py has cleaner messages

### Difference F3 — BUG in main2.py `run_expanding_window_evaluation` (line ~2811)

```python
# main2.py line ~2811 — BUG: uses "rolling" path for expanding window combo file
combo_path = os.path.join(paths.iter_dir("rolling"), ...)
```
Should be `paths.iter_dir("expanding")`. This causes expanding window combo artifacts to be written into the rolling window output directory, potentially overwriting rolling window files or creating confusion. main.py does not have this bug.

### Difference F4 — `run_comprehensive_evaluation` Return Type and Resume Support

main.py returns `(all_results, completed_evaluations)` — a tuple. main2.py returns only `all_results`. Any caller that expects the tuple form (as in main.py's `enhanced_main`) will break if used with main2.py's version.

### Difference F5 — `generate_evaluation_summary` Crash Risk

main2.py accesses `holdout['filtered_results']` with direct dict indexing. If that key is absent (e.g., holdout evaluation was skipped or produced no filtered results), main2.py raises a `KeyError`. main.py uses `.get('filtered_results', pd.DataFrame())` and is safe.

### Difference F6 — `enhanced_main()` Unreachable Dead Loop in main2.py

main2.py contains a second `while True:` loop starting at line 4477 that is structurally unreachable — the first loop breaks before it, and there is no code path that reaches the second one. This is a copy-paste artifact and does not affect runtime behavior but signals the file was assembled from multiple drafts without cleanup.

### Difference F7 — `normalize_price_panel` Call in main2.py

main2.py calls `normalize_price_panel(full_price_data)` before signal generation (inside the cache-miss branch). main.py does not. This is a behavioral difference: if `normalize_price_panel` modifies prices, the signals generated will differ.

---

## 7. Verdict: Which File is the Reference Implementation?

**Recommendation: main.py is the more correct reference implementation for production use.**

Reasoning:

1. **main.py has the resume/checkpoint system** (`completed_evaluations`, master checkpoint save/load). This is critical for long multi-hour runs. main2.py silently dropped this entire feature, which represents a significant regression in operational robustness.

2. **main.py does not contain the expanding-window path bug** (Difference F3). main2.py writes expanding window combo artifacts to the rolling window directory — a data-corruption bug.

3. **main.py's `generate_evaluation_summary` uses safe dict access** and will not crash on partial results. main2.py will raise `KeyError` if holdout results are absent.

4. **main.py's `_run_single_backtest` is simpler and cleaner** — precondition filtering is done before the parallel pool is created, reducing per-task data size.

5. **main2.py has a dead unreachable loop** (lines 4477–4499) confirming it was assembled from multiple drafts and not cleaned up.

**However**, main2.py has two meaningful improvements that should be cherry-picked into main.py:

- **The `precond_mask` pass-through architecture** (Differences F1/F2) is more semantically correct for time-windowed evaluations, where the mask should be applied per window, not used to pre-filter the signal universe globally.
- **The `max_workers` increases** (10→12 for downloads, 5→6 for backtesting/combos) are straightforward tuning improvements.
- **The `_tqdm()` wrapping of iteration loops** in the three runner functions improves observability.
- **The `write_manifest`/`write_readme` calls** in `enhanced_main` are useful for run documentation.

The single most important difference is **Difference F3** (the expanding window path bug in main2.py). If main2.py were used for a run that includes both rolling and expanding window evaluations, expanding window combo files would silently overwrite rolling window files, corrupting both datasets with no error message.
