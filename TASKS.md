# Signal Pipeline — Task List

**Working directory:** `signal_pipeline/`  
**Reference:** `PLANNING.md` (full specifications, architecture diagrams, and implementation details for every task below)  
**Existing code for reference:** `composer_signal_generator/main.py` (reference; `main2.py` has a confirmed bug and should be deleted after back-porting two items — see PLANNING.md)  
**Execution:** Work through tasks in order. Dependencies are strict — do not skip phases. Mark each `[x]` when complete and verified.

---

## Phase 1 — Project Bootstrap and Data Layer

- [ ] **1.1** Create the source directory tree.  
  Create: `src/`, `src/data/`, `src/__init__.py`, `src/data/__init__.py`  
  Create empty placeholder files: `src/indicators.py`, `src/signals.py`, `src/backtest.py`, `src/metrics.py`, `src/combos.py`, `src/validation.py`, `src/monte_carlo.py`, `src/composer.py`, `src/report.py`  
  Create: `config.py`, `main.py`, `rsi_search.py`, `analysis_workshop.py`, `requirements.txt`, `.env.example`  
  **Verify:** All paths exist.

- [ ] **1.2** Write `requirements.txt`.  
  Contents: `numpy`, `pandas`, `scipy`, `requests`, `python-dotenv`, `matplotlib`, `seaborn`, `tqdm`, `composer-tools` (optional, with graceful fallback).  
  Do NOT include `quantstats` — all metrics will be implemented natively.

- [ ] **1.3** Port the data layer.  
  Copy `composer_signal_generator/../strategy_viewer/fuzz_tester/strategy_engine/src/config_loader.py` → `src/data/config_loader.py`.  
  Copy `strategy_viewer/fuzz_tester/strategy_engine/src/data_loader.py` → `src/data/loader.py`.  
  Copy `strategy_viewer/fuzz_tester/strategy_engine/src/data_alignment.py` → `src/data/alignment.py`.  
  In `src/data/config_loader.py`: remove the `config_dict` passthrough parameter — simplify to return `api_keys` list only.  
  In `src/data/alignment.py`: add function `load_multi_ticker_aligned(tickers: list[str], data_dir: Path) -> pd.DataFrame` that loads all tickers, aligns on a common date index, forward-fills, and returns a single DataFrame with tickers as columns.  
  No other changes to these files.  
  **Verify:** `python -c "from src.data.loader import check_freshness_and_update; print('ok')"` succeeds.

- [ ] **1.4** Write `src/indicators.py`.  
  Port from `strategy_viewer/fuzz_tester/strategy_engine/src/indicators.py`:  
  — `calculate_sma(series, period) -> pd.Series`  
  — `calculate_ema(series, period) -> pd.Series`  
  — `calculate_rsi(series, period) -> pd.Series` (Wilder's smoothing — preserve exact implementation)  
  — `calculate_cumret(series, period) -> pd.Series`  
  Port from `strategy_viewer/fuzz_tester/fuzz_tester.py`:  
  — `calculate_maxdd(series, period) -> pd.Series` (rolling peak-to-trough, strict order-aware)  
  — `calculate_mareturn(series, period) -> pd.Series` (rolling mean of daily returns)  
  Add dispatch function:  
  `compute_indicator(series: pd.Series, fn_label: str, window: int) -> pd.Series`  
  that maps `fn_label` strings ("RSI", "SMA", "EMA", "CumRet", "MaxDD", "MAReturn", "Price") to the appropriate function.  
  **Verify:** Unit test: compute RSI(SPY, 14) from a known CSV and confirm values are in 0–100 range with NaN for first 14 rows.

- [ ] **1.5** Build the indicator cache.  
  In `src/indicators.py`, add:  
  ```python
  def build_indicator_cache(price_df: pd.DataFrame,
                             required: list[tuple[str, str, int]]) -> dict:
  ```  
  `required` is a list of `(ticker, fn_label, window)` tuples.  
  Returns `dict` keyed by `(ticker, fn_label, window)` → `np.ndarray` (float64, shape `(n_days,)`).  
  Each value is `compute_indicator(price_df[ticker], fn_label, window).to_numpy(dtype=float)`.  
  Deduplicate: if the same `(ticker, fn_label, window)` appears multiple times in `required`, compute it only once.  
  **Verify:** Calling with two identical entries in `required` results in one computation, not two (add a counter or log to confirm).

---

## Phase 2 — Signal Matrix Generation

- [ ] **2.1** Write `src/signals.py` — signal naming.  
  Implement `make_signal_name(lhs_ticker, lhs_fn, lhs_window, comparator, rhs_type, rhs_value=None, rhs_ticker=None, rhs_fn=None, rhs_window=None) -> str`.  
  Naming convention (see `PLANNING.md` §2.3):  
  — Fixed: `{fn}_{window}_{ticker}_{COMPARATOR}_{threshold}` e.g. `RSI_10_SPY_GT_50`  
  — Indicator vs indicator: `{fn}_{window}_{ticker}_{COMPARATOR}_{rhs_fn}_{rhs_window}_{rhs_ticker}` e.g. `SMA_20_QQQ_GT_SMA_20_TLT`  
  Comparator mapping: `{"gt": "GT", "lt": "LT", "gte": "GTE", "lte": "LTE"}`.  
  This naming is the interface contract for Composer export. Do not change it after this step.

- [ ] **2.2** Write `src/signals.py` — signal metadata structure.  
  Define `SignalSpec` as a dataclass or TypedDict with fields:  
  `name, lhs_ticker, lhs_fn, lhs_window, comparator, rhs_type, rhs_value, rhs_ticker, rhs_fn, rhs_window, target`.  
  Implement `derive_required_indicators(signal_specs: list[SignalSpec]) -> list[tuple[str,str,int]]`  
  that extracts all unique `(ticker, fn, window)` pairs needed by the signal list.

- [ ] **2.3** Write `src/signals.py` — signal generation from config.  
  Implement `generate_signal_specs(config: dict) -> list[SignalSpec]`.  
  Config keys (see `PLANNING.md` §2.2):  
  — `signal_tickers`: list of tickers to use as LHS  
  — `target_tickers`: list of allocation tickers  
  — `rsi_windows`: list of int  
  — `rsi_thresholds`: list of float  
  — `rsi_comparators`: list of str ("lt", "gt")  
  — `sma_windows`: list of int  
  — `ema_windows`: list of int  
  — `cross_tickers`: list of tickers allowed on RHS for indicator-vs-indicator signals  
  Generate signal specs for each configured signal type. Each (signal, target) pair is a separate spec.

- [ ] **2.4** Write `src/signals.py` — signal matrix builder.  
  Implement `generate_signal_matrix(specs: list[SignalSpec], indicator_cache: dict, date_index: np.ndarray) -> tuple[np.ndarray, list[str], list[SignalSpec]]`.  
  Returns `(signal_matrix, signal_names, signal_metadata)` where:  
  — `signal_matrix`: `np.ndarray[bool]` shape `(n_days, n_signals)`  
  — `signal_names`: `list[str]` parallel to columns  
  — `signal_metadata`: `list[SignalSpec]` parallel to columns  
  For each spec: look up LHS and RHS arrays from `indicator_cache`, apply comparator, produce boolean column. NaN days → False.  
  Stack all columns: `np.column_stack(boolean_arrays).astype(bool)`.  
  **Verify:** For a small test config (3 tickers, 2 targets, RSI only), print `signal_matrix.shape` and confirm it matches expected `(n_days, n_specs × n_targets)`.

---

## Phase 3 — Backtesting Engine and Metrics

- [ ] **3.1** Write `src/metrics.py` — all metrics in pure NumPy.  
  No quantstats dependency. Implement each metric operating on a 1D return array `r: np.ndarray`:  
  — `sharpe(r, annual=252)`: `mean(r) / std(r) * sqrt(annual)`  
  — `smart_sharpe(r, annual=252)`: Sharpe corrected for autocorrelation (sum of first 5 autocorrelations). Formula in `PLANNING.md` §3.3.  
  — `sortino(r, annual=252)`: `mean(r) / std(r[r<0]) * sqrt(annual)`. Return 0 if no negative days.  
  — `max_drawdown(r)`: max peak-to-trough from cumulative log return curve. Return as positive %.  
  — `calmar(r, annual=252)`: `cagr(r, annual) / max_drawdown(r)`. Guard against zero drawdown.  
  — `omega(r, threshold=0.0)`: `sum(max(r-threshold, 0)) / sum(max(threshold-r, 0))`. Return `inf` if no losses.  
  — `win_rate(r)`: `(r > 0).mean()`  
  — `profit_factor(r)`: `sum(r[r>0]) / abs(sum(r[r<0]))`. Return `inf` if no losses.  
  — `recovery_factor(r)`: `total_return(r) / max_drawdown(r)`  
  — `cagr(r, annual=252)`: `(1 + r).prod() ** (annual / len(r)) - 1`  
  — `total_return(r)`: `(1 + r).prod() - 1`  
  Then implement batch versions operating on `(n_days, n_signals)` matrices — vectorized across columns.  
  **Verify:** Compute `sharpe`, `sortino`, `calmar` on a known return series and compare to hand-calculated values.

- [ ] **3.2** Write `src/metrics.py` — tail event metrics.  
  Implement `tail_metrics(r: np.ndarray) -> dict` for a single signal's return array.  
  Full implementation in `PLANNING.md` §3.3 (Tail Event Metrics table).  
  Returns: `tail_score, tail_concentration, excess_kurtosis, base_win_rate, stripped_win_rate, wr_delta`.  
  For batch computation across many signals, iterate in a plain loop (kurtosis is not vectorizable; this runs once at base parameters, not in the sweep inner loop).

- [ ] **3.3** Write `src/backtest.py` — MOC return preparation.  
  Implement `prepare_moc_returns(raw_returns: np.ndarray) -> np.ndarray`.  
  `np.roll(raw_returns, -1)` then set last element to 0.0.  
  This shift must be applied exactly once before any backtest call. Document in the function docstring that this implements Composer's verified execution model (signal at close t → return from close t to close t+1). Do NOT add a NEXT_BAR mode.

- [ ] **3.4** Write `src/backtest.py` — vectorized batch backtest.  
  Implement `batch_backtest(signal_matrix, target_returns_moc, bil_returns) -> dict[str, np.ndarray]`.  
  Full implementation in `PLANNING.md` §3.1.  
  Returns a dict where each key is a metric name and each value is a `(n_signals,)` array.  
  All computation via NumPy broadcasting — no Python loops over signals.  
  **Verify:** Run on a 7500×315 signal matrix. Confirm output shape is `(315,)` per metric. Time it — should complete in under 5 seconds.

- [ ] **3.5** Write `src/backtest.py` — process pool infrastructure.  
  Implement the worker initializer pattern (see `PLANNING.md` §3.2):  
  Module-level globals: `_SIGNAL_MATRIX`, `_TARGET_RETURNS`, `_BIL_RETURNS`, `_DATE_INDEX`.  
  `_init_worker(signal_matrix, target_returns, bil_returns, date_index)` populates them.  
  `_backtest_window(window_spec: dict) -> dict` slices globals by date index and calls `batch_backtest`.  
  `run_parallel_backtests(window_specs, signal_matrix, target_returns, bil_returns, date_index, n_workers=None)` manages the pool.  
  **Verify:** Run `run_parallel_backtests` with 4 windows. Confirm results are identical to running `batch_backtest` directly on each slice.

---

## Phase 4 — Combo Generation and Backtesting

- [ ] **4.1** Write `src/combos.py` — combo name parsing.  
  Implement `parse_combo_name(name: str) -> tuple[list[str], list[str]]`.  
  Splits a combo signal name into `(members, operators)` using the known operator set `{"AND", "OR", "A_AND_NOT_B", "B_AND_NOT_A"}` as delimiters. See `PLANNING.md` §8.1 for the correct implementation.  
  This is the fix for the existing code's bug of splitting on `+` indiscriminately.  
  **Verify:** `parse_combo_name("RSI_10_SPY_GT_50+AND+SMA_20_QQQ_GT_SMA_20_TLT")` returns `(["RSI_10_SPY_GT_50", "SMA_20_QQQ_GT_SMA_20_TLT"], ["AND"])`.

- [ ] **4.2** Write `src/combos.py` — combo column generation.  
  Implement `apply_operator(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray` for all four operators.  
  Implement `build_combo_batch(signal_matrix, batch: list[tuple[int,int,str]]) -> np.ndarray`  
  that takes a list of `(i, j, operator)` tuples and returns a `(n_days, batch_size)` boolean matrix.  
  Release the batch matrix from memory after metrics are computed — do not accumulate.

- [ ] **4.3** Write `src/combos.py` — batched combo backtester.  
  Implement `run_combo_backtests(signal_matrix, signal_names, signal_metadata, target_returns_dict, bil_returns, date_index, top_k_for_combos, config) -> list[dict]`.  
  Full implementation in `PLANNING.md` §4.3.  
  `COMBO_BATCH_SIZE = 10_000` (tunable in config).  
  Pool selection for combos: use top-K signals by individual OOS Sortino (configurable, default 500). This is a combinatorial feasibility cap, NOT a quality filter — document clearly in code comments.  
  For each batch: build combo columns, run `batch_backtest`, store results, del combo matrix, continue.  
  **Verify:** Run on a 315-signal matrix with top-100 pool. Confirm no OOM, results list has expected length (C(100,2) × 4 = 19,800 per target).

---

## Phase 5 — Validation Framework

- [ ] **5.1** Write `src/validation.py` — window slicing.  
  Implement `generate_walk_forward_windows(n_days, train_size, test_size) -> list[dict]`.  
  Implement `generate_expanding_windows(n_days, initial_train, test_size) -> list[dict]`.  
  Implement `generate_rolling_windows(n_days, train_size, test_size, step) -> list[dict]`.  
  Each returns a list of `{"train_start", "train_end", "test_start", "test_end"}` index dicts.  
  All window generation is pure index arithmetic — no data access.

- [ ] **5.2** Write `src/validation.py` — per-window evaluation.  
  Implement `run_validation(signal_matrix, signal_names, signal_metadata, price_df, target_tickers, bil_returns, window_type, window_config, n_workers) -> pd.DataFrame`.  
  For each window: slice signal_matrix and return arrays by test indices, call `run_parallel_backtests`, collect results.  
  Each row in the output DataFrame: one (signal, target, window_iteration) result with all metrics.  
  Include `window_iteration`, `test_start_date`, `test_end_date` columns.

- [ ] **5.3** Write `src/validation.py` — OOS aggregation.  
  Implement `aggregate_oos_results(results_df: pd.DataFrame) -> pd.DataFrame`.  
  Groups by `(signal_name, target)`.  
  For each group computes: `Sharpe_p10`, `Sharpe_p50`, `Sharpe_p90`, `Sharpe_IQR`, `Return_p50`, `Return_p10`, `MaxDD_p90`, `Consistency_Score` (fraction of windows with positive Sharpe), `N_Iterations`.  
  Returns one row per (signal, target) — this is the primary output DataFrame.  
  **Verify:** Run on a small test case (3 signals, 2 targets, 5 windows). Output DataFrame has 6 rows (3 × 2), all aggregation columns present.

---

## Phase 6 — Monte Carlo Integration

- [ ] **6.1** Port the Monte Carlo engine.  
  Copy the following functions from `monte_carlo_sim/monte_carlo_sim/Monte Carlo walk forward composer working.py` into `src/monte_carlo.py`:  
  — `run_monte_carlo_simulation(returns, num_simulations, simulation_length, annual_periods)`  
  — `analyze_drawdowns(returns, output_dir, period_length, test_start_date, test_end_date, portfolio_name, dates)`  
  — `run_walk_forward_test(dates, returns, test_period_length, output_dir, portfolio_name)`  
  — `plot_drawdown_distributions(simulation_results, actual_max_drawdown, actual_dd_duration, period_length, output_dir, portfolio_name)`  
  Remove `fetch_backtest()` (hits Composer API — not needed here; return series come from validation).  
  Remove `calculate_portfolio_returns()` (replaced by backtest engine).  
  Keep all charting and statistics logic unchanged.

- [ ] **6.2** Write `src/monte_carlo.py` — pipeline interface.  
  Implement `run_mc_for_signal(signal_name, oos_returns, dates, output_dir) -> dict`.  
  `oos_returns`: the actual OOS daily return series from walk-forward validation.  
  Runs `run_walk_forward_test` for 63-day, 126-day, and 252-day horizons.  
  Writes PNG charts to `output_dir/monte_carlo/{signal_name}/`.  
  Returns summary stats dict.

- [ ] **6.3** Write `src/monte_carlo.py` — portfolio MC.  
  Implement `run_mc_for_portfolio(top_n_signal_names, oos_returns_dict, dates, output_dir) -> dict`.  
  Constructs equal-weight portfolio: `portfolio_returns = np.mean(np.column_stack([oos_returns_dict[n] for n in top_n_signal_names]), axis=1)`.  
  Runs same MC suite as per-signal.  
  Writes to `output_dir/monte_carlo/combined_portfolio/`.

---

## Phase 7 — RSI Search Entry Point

- [ ] **7.1** Write `rsi_search.py`.  
  Full specification in `PLANNING.md` §7.  
  Config keys: `signal_tickers`, `target_tickers`, `rsi_windows`, `rsi_thresholds`, `comparators`, `benchmark_tickers`, `min_trades`, `min_win_rate`.  
  Uses: `src/data/` (Tiingo download), `src/indicators.py` (RSI), `src/backtest.py` (batch_backtest), `src/metrics.py`.  
  Does NOT use combos, validation windows, or Monte Carlo.  
  Output: CSV to `output/rsi_search_{timestamp}.csv`.  
  Columns: `Signal`, `Target`, `Win_Rate`, `N_Trades`, `Benchmark_Median_Return`, `Total_Return`, `Sharpe`, `Tail_Concentration`, `Best_Target_IS` (in-sample best target, labelled as such).  
  **Verify:** Run against 3 signal tickers, 2 targets, 2 RSI windows, completes in under 2 minutes, output CSV is non-empty.

---

## Phase 8 — Composer Export

- [ ] **8.1** Write `src/composer.py` — signal name parser.  
  Implement `parse_signal_name(name: str) -> dict` that decomposes a structured signal name back into its component fields (`lhs_fn`, `lhs_window`, `lhs_ticker`, `comparator`, `rhs_type`, `rhs_value` or `rhs_fn/rhs_window/rhs_ticker`).  
  Use the naming convention established in Task 2.1 — parser and naming are inverses of each other.  
  **Verify:** `parse_signal_name(make_signal_name(...))` round-trips cleanly for all signal types.

- [ ] **8.2** Write `src/composer.py` — condition node builder.  
  Implement `signal_to_composer_condition(parsed: dict) -> dict`.  
  Maps parsed signal fields to a Composer JSON condition node (lhs, rhs, comparator keys per Composer schema).  
  `fn_map` from Python label to Composer function name: `{"RSI": "relative-strength-index", "SMA": "moving-average-price", "EMA": "exponential-moving-average-price", ...}`.

- [ ] **8.3** Write `src/composer.py` — combo condition builder.  
  Implement `combo_to_composer_condition(combo_name: str) -> dict`.  
  Uses `parse_combo_name()` (from Task 4.1) to split into members and operators.  
  Builds a nested Composer `compound` condition with `operator: "all"` for AND, `operator: "any"` for OR.  
  `A_AND_NOT_B` and `B_AND_NOT_A` require explicit nesting (A AND NOT B = compound{all: [A, NOT{B}]}).

- [ ] **8.4** Write `src/composer.py` — precondition translation.  
  Implement `precond_expr_to_composer_condition(expr: str, price_df: pd.DataFrame) -> dict`.  
  Input: a precondition string like `"PRICE('SPY') > SMA('SPY', 200)"`.  
  Parse using the same grammar as `_safe_eval_precond` (from `composer_signal_generator/main2.py`) but output a Composer condition node instead of evaluating to a boolean Series.  
  The precondition and signal must be combined into a `compound{operator: "all"}` node when generating the symphony.

- [ ] **8.5** Write `src/composer.py` — symphony assembler.  
  Implement `build_symphony(top_n_specs: list[dict], safe_asset: str = "BIL") -> dict`.  
  Each entry in `top_n_specs`: `{signal_name, target_ticker, preconditions}`.  
  Builds the full Composer symphony JSON tree (see `PLANNING.md` §8.5 for structure).  
  Each signal gets one `if` block with a `wt-cash-equal` weight.  
  Returns the dict — caller writes it to JSON.

- [ ] **8.6** Write `src/composer.py` — round-trip verification.  
  Implement `verify_composer_output(symphony_json: dict, signal_matrix: np.ndarray, signal_names: list[str], price_df: pd.DataFrame) -> dict`.  
  Re-parses the generated symphony JSON using `extract_conditions_from_tree` logic (can be a simplified inline version — no need to import from the strategy viewer).  
  Re-evaluates conditions against `price_df`.  
  For each signal: compares re-evaluated boolean array to the original `signal_matrix` column.  
  Returns `{signal_name: {"match_rate": float, "warning": bool}}`.  
  Log a warning if any signal has match rate < 99%.  
  **Verify:** Build a symphony for 2 known signals. Verify match rate is 100%.

---

## Phase 9 — Report Generation

- [ ] **9.1** Write `src/report.py` — CSV outputs.  
  Implement `write_output_csvs(all_signals_df, all_combos_df, top_n_df, output_dir)`.  
  Writes: `all_signals.csv`, `all_combos.csv`, `top_n_signals.csv` to `output/{timestamp}/`.  
  All DataFrames include full metrics columns from aggregation (Phase 5.3) plus tail event columns (Phase 3.2).

- [ ] **9.2** Write `src/report.py` — summary HTML dashboard.  
  Implement `build_html_report(top_n_df, portfolio_mc_summary, config, output_dir) -> str`.  
  Self-contained HTML file (no external dependencies, no server required).  
  Contents: run config summary, sortable top-N signals table with all metrics, tail event badges per signal (same colour scheme as Strategy Viewer), combined portfolio equity curve vs BIL, links to individual MC report PNGs.  
  Write to `output/{timestamp}/report.html`.

- [ ] **9.3** Port `analysis_workshop.py`.  
  Copy `composer_signal_generator/analysis_workshop.py` into the project root as `analysis_workshop.py`.  
  Add: auto-discovery of the most recent `output/` subdirectory (prompt user to confirm or select).  
  Add to filter menu: `Omega_Ratio` (compute on load from return series if not in CSV), `Tail_Concentration`, `Excess_Kurtosis`, `Stripped_Win_Rate`.  
  No other changes to the core filtering logic.

---

## Phase 10 — Main Orchestrator

- [ ] **10.1** Write `config.py` — default configuration.  
  All tunable parameters with documented defaults:  
  `COMBO_BATCH_SIZE = 10_000`, `COMBO_POOL_TOP_K = 500`, `N_WORKERS = cpu_count - 1`,  
  `WF_TRAIN_DAYS = 504`, `WF_TEST_DAYS = 63`, `MIN_SIGNAL_DAYS = 20`,  
  `TOP_N_SIGNALS = 10`, `MC_NUM_SIMULATIONS = 10_000`,  
  `SAFE_ASSET = "BIL"`, `DEFAULT_SORT_METRIC = "sortino"`.

- [ ] **10.2** Write `main.py` — user prompts.  
  Interactive prompt sequence:  
  — Signal tickers (comma-separated)  
  — Target tickers (comma-separated)  
  — Metrics to include (RSI / SMA / EMA / all)  
  — RSI windows and thresholds  
  — SMA/EMA windows  
  — Validation method (walk-forward / expanding / rolling / all)  
  — Top-N count and sort metric (from the 8 options in `PLANNING.md` §3.3)  
  — Run combos? (y/n)  
  — Preconditions (optional, press Enter to skip)  
  — Output directory  
  Validate inputs before proceeding. Print a summary and ask for confirmation before starting.

- [ ] **10.3** Write `main.py` — pipeline orchestration.  
  Call sequence (see `PLANNING.md` Data Flow diagram):  
  1. Load config + API keys  
  2. Download/refresh price data (`check_freshness_and_update`)  
  3. Load aligned price DataFrame  
  4. Generate signal specs from config  
  5. Build indicator cache  
  6. Generate signal matrix  
  7. Run validation (all selected methods)  
  8. Aggregate OOS results  
  9. If combos enabled: run combo backtests, aggregate  
  10. Select top-N by chosen metric  
  11. Compute tail metrics for top-N  
  12. Run Monte Carlo for top-N signals + portfolio  
  13. Build Composer symphony  
  14. Run round-trip verification  
  15. Write all outputs (CSVs, HTML report, MC charts, symphony JSON)  
  16. Print completion summary with output path  
  Each step prints a progress header. Wrap in try/except — on failure, print which step failed and re-raise.  
  **Verify:** Full run on minimal config (3 signal tickers, 2 targets, RSI only, no combos). Completes without exception. All output files written.

---

## Phase 11 — Integration Testing

- [ ] **11.1** Benchmark test — individual signals.  
  Config: 15 signal tickers, 4 target assets, RSI (3 windows × 3 thresholds), SMA (9 windows), EMA (9 windows). No combos.  
  Run `main.py`. Confirm: completes without OOM, runtime under 30 minutes, `all_signals.csv` is non-empty, `top_n_signals.csv` has N rows, `report.html` opens in browser.

- [ ] **11.2** Benchmark test — with combos.  
  Same config as 11.1, combos enabled with `COMBO_POOL_TOP_K = 500`.  
  Confirm: completes without OOM, runtime under 4 hours, `all_combos.csv` is non-empty.

- [ ] **11.3** Composer round-trip test.  
  Take the top-3 signals from 11.1. Run `verify_composer_output`. Confirm all three have match rate ≥ 99%.  
  Manually import `symphony.json` into Composer.Trade. Confirm it loads without error.

- [ ] **11.4** RSI Search smoke test.  
  Run `rsi_search.py` with 5 signal tickers, 2 targets. Confirm: completes in under 5 minutes, output CSV has `Best_Target_IS` column, filtered rows satisfy `min_trades` and `min_win_rate` thresholds.

- [ ] **11.5** Analysis Workshop test.  
  Run `analysis_workshop.py` pointing at the 11.1 output directory.  
  Confirm: all tail event columns appear in the filter menu, Omega Ratio is computed and filterable, saving the filtered output produces a valid CSV.
