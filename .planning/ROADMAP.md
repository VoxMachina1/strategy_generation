# Roadmap: Composer Signal Pipeline

**Project:** Composer Signal Pipeline  
**Milestone:** v1.0 — Full Discovery → Validation → Assembly Pipeline  
**Granularity:** standard  
**Vision:** See `VISION.md` | **Detailed specs:** See `PLANNING.md`

---

## Phases

- [x] **Phase 0: Pre-Requisite Bug Fixes** - Fix all confirmed bugs in the existing codebase before the Phase 1 rewrite begins
- [x] **Phase 1: Project Bootstrap and Data Layer** - Create the new src/ directory structure and port the working data layer
- [x] **Phase 2: Signal Matrix Generation** - Replace dict signal store with boolean numpy signal matrix
- [x] **Phase 3: Vectorized Backtesting Engine** - Implement core vectorized backtest and process pool with initializer pattern
- [x] **Phase 4: Combo Generation and Backtesting** - Implement batched pairwise combo generation and backtesting
- [x] **Phase 5: Validation Framework** - Walk-forward evaluation with crisis hold-out, stripped Sharpe, and regime analysis
- [x] **Phase 5b: Tail Analysis Automation** - Port compute_tail_metrics() and automate for all validated signals
- [x] **Phase 6: Monte Carlo Integration** - Port MC simulation and wire to pipeline output
- [x] **Phase 7: RSI Search Entry Point** - Implement fast rsi_search.py entry point
- [x] **Phase 8: Composer Export** - Fix signal name parser, Type 1 leaf insertion, Type 2 root wrapping, Mode C
- [x] **Phase 9: Output and Report Generation** - CSV outputs, HTML dashboard, Composer JSON, Analysis Workshop port

---

## Phase Details

### Phase 0: Pre-Requisite Bug Fixes

**Goal:** All confirmed bugs in the existing codebase are resolved so the Phase 1 rewrite starts from a known-good baseline  
**Depends on:** Nothing  
**Success Criteria** (what must be TRUE):

  1. The `freeze_combo_universe` dead code block (lines ~3084–3141 of `composer_signal_generator/main.py`) is removed — no calls to undefined functions remain
  2. `precond_mask` is correctly threaded through all runner functions in `main.py` (back-ported from `main2.py`)
  3. `max_workers` is user-configurable via an interactive prompt in `main.py`
  4. `main2.py` is deleted
  5. `_ATR()` is renamed to `_rolling_std()` in `main.py`; a new `_ATR()` implementing true Wilder ATR is added
  6. `_BBAND_LOWER()` is added to `main.py`
  7. `rsi_tester/strategy_engine/src/data_loader.py` fetches `adjOpen`, `adjHigh`, `adjLow`, `adjClose`; new CSV format is `date,open,high,low,close`; old `date,close` CSVs load correctly (backward compat)
  8. `data_alignment.py` and `load_ticker_csv()` handle OHLC columns
  9. `calculate_atr()` (Wilder ATR) and `calculate_bbands_lower()` exist in `rsi_tester/strategy_engine/src/indicators.py`
  10. Monte Carlo `plot_drawdown_distributions()` receives simulation `max_drawdowns` in percent (not fractions)
  11. `rsi_tester/strategy_engine/src/metrics.py` counts signal 0→1 transitions for `Total_Trades`, not active days

**Plans:** TBD

### Phase 1: Project Bootstrap and Data Layer

**Goal:** The `signal_pipeline/src/` directory tree exists and the Tiingo data layer, indicators, and indicator cache are ported and passing basic verification  
**Depends on:** Phase 0  
**Success Criteria** (what must be TRUE):

  1. All directories and placeholder files from the planned structure exist
  2. `python -c "from src.data.loader import check_freshness_and_update; print('ok')"` succeeds
  3. RSI(SPY, 14) computed from a cached CSV returns values in 0–100 range with NaN for the first 14 rows
  4. `build_indicator_cache()` with duplicate `(ticker, fn, window)` entries computes each unique key only once

**Plans:** TBD

### Phase 2: Signal Matrix Generation

**Goal:** Signal generation produces a `(n_days × n_signals)` boolean numpy matrix with correct naming convention  
**Depends on:** Phase 1  
**Success Criteria** (what must be TRUE):

  1. `generate_signal_matrix()` returns `(signal_matrix, signal_names, signal_metadata, date_index)` with correct shapes
  2. Signal names follow `{fn}_{window}_{ticker}_{comparator}_{threshold_or_rhs}` format parseable by the export layer
  3. NaN warmup days are set to `False` (not NaN) in the signal matrix
  4. `parse_signal_name()` round-trips correctly for all supported indicator types

**Plans:** TBD

### Phase 3: Vectorized Backtesting Engine

**Goal:** `batch_backtest()` correctly computes all metrics for all signals simultaneously using vectorized numpy operations, with no IPC data-copying per task  
**Depends on:** Phase 2  
**Success Criteria** (what must be TRUE):

  1. `batch_backtest()` returns the correct Sharpe, Sortino, and Win Rate for a known signal against a known return series (verified against a reference scalar calculation)
  2. Worker processes receive the shared data arrays via the initializer, not as task arguments
  3. All 13 specified metrics (Total Return, CAGR, Sharpe, Smart Sharpe, Sortino, Calmar, Omega, Win Rate, Profit Factor, Recovery Factor, Max Drawdown, Time in Market, N Signal Days) are computed correctly

**Plans:** TBD

### Phase 4: Combo Generation and Backtesting

**Goal:** All pairwise signal combinations are generated and backtested in memory-bounded batches without ever materializing the full combo matrix  
**Depends on:** Phase 3  
**Success Criteria** (what must be TRUE):

  1. Peak memory usage during combo backtesting stays below `COMBO_BATCH_SIZE × n_days × 1 byte + overhead` (verified with a memory profiler or RSS check)
  2. Combo names are correctly parsed by `parse_combo_name()` — operator tokens are not confused with signal name tokens
  3. All four operators (AND, OR, A_AND_NOT_B, B_AND_NOT_A) produce correct boolean combinations

**Plans:** TBD

### Phase 5: Validation Framework

**Goal:** Walk-forward, expanding window, and rolling window evaluation run correctly; crisis hold-out, stripped Sharpe, and regime stats are computed and present in the output CSV  
**Depends on:** Phase 4  
**Success Criteria** (what must be TRUE):

  1. OOS windows are non-overlapping for walk-forward mode
  2. `HitRate_Positive_Sharpe`, `Sharpe_CoV`, `Sharpe_Stripped`, `Crisis_Sharpe` columns are present in the results CSV
  3. A signal that never fires during a crisis epoch has `Crisis_Sharpe = NaN` (not a fail)
  4. `Regime_Count`, `Regime_Duration_Median`, `Regime_Hit_Rate`, `Signal_Type` columns are present
  5. `Signal_Type` is `"Type1"` for short-duration signals and `"Type2"` for long-duration signals per the configured threshold

**Plans:** TBD

### Phase 5b: Tail Analysis Automation

**Goal:** Tail analysis metrics are computed automatically for every validated signal and included in the output CSV  
**Depends on:** Phase 5  
**Success Criteria** (what must be TRUE):

  1. `Tail_Concentration`, `Excess_Kurtosis`, `Base_Win_Rate`, `Stripped_Win_Rate`, `WR_Delta`, `Tail_Score` columns are present in the results CSV
  2. Tail metrics match Strategy Viewer's `compute_tail_metrics()` output for the same return series (verified by cross-check)
  3. `Combined_Score` column exists: `0.6 × oos_quality_score + 0.4 × tail_score`

**Plans:** TBD

### Phase 6: Monte Carlo Integration

**Goal:** Top-N signals and the combined portfolio each have a Monte Carlo report  
**Depends on:** Phase 5b  
**Success Criteria** (what must be TRUE):

  1. Per-signal MC fan chart PNGs are written for all top-N signals
  2. Portfolio MC report is written using equal-weight combination of top-N OOS return series
  3. The drawdown unit mismatch bug is NOT present (simulation max_drawdowns and actual max_drawdown are both in percent when passed to `plot_drawdown_distributions()`)

**Plans:** TBD

### Phase 7: RSI Search Entry Point

**Goal:** `rsi_search.py` runs in under 5 minutes for a 15-ticker, 4-target config and produces a correct filtered CSV  
**Depends on:** Phase 3  
**Success Criteria** (what must be TRUE):

  1. `python rsi_search.py` completes for a 15-ticker, 4-target config in under 5 minutes
  2. Output CSV includes `Signal`, `Target`, `Win_Rate`, `N_Trades`, `Benchmark_Median_Return`, `Total_Return`, `Sharpe`, `Tail_Concentration`, `Best_Target_IS`
  3. `benchmark_median_return < 0` is flagged as a column, configurable as a filter

**Plans:** TBD

### Phase 8: Composer Export

**Goal:** Type 1 (leaf) and Type 2 (root-wrap) assembly produce valid Composer JSON that round-trips correctly, and Mode C (extend existing strategy) works end-to-end  
**Depends on:** Phase 5b  
**Success Criteria** (what must be TRUE):

  1. `parse_combo_name()` correctly separates member signals from operator tokens for all combo types
  2. `output/{timestamp}/symphony.json` is valid importable Composer JSON
  3. Round-trip verification passes with < 1% signal disagreement for all top-N signals
  4. A signal backtested with a precondition produces Composer JSON that includes the precondition as a nested condition node
  5. Mode C: loading an existing strategy JSON and running discovery correctly inserts Type 1 signals at leaf nodes and Type 2 signals as root wrappers

**Plans:** TBD

### Phase 9: Output and Report Generation

**Goal:** All CSV, PNG, HTML, and JSON outputs are written to `output/{timestamp}/` and the Analysis Workshop loads them correctly  
**Depends on:** Phase 8  
**Success Criteria** (what must be TRUE):

  1. `all_signals.csv`, `top_n_signals.csv`, `symphony.json`, `report.html` all exist in the output directory after a full run
  2. `report.html` opens in a browser and the top-N signals table is sortable by any metric column
  3. `analysis_workshop.py` loads `top_n_signals.csv` and all tail event columns are filterable
  4. The combined portfolio equity curve vs BIL is visible in `report.html`

**Plans:** TBD

---

## Progress

| Phase | Plans Complete | Status |
|-------|----------------|--------|
| 0. Pre-Requisite Bug Fixes | 1/1 | Complete ✓ |
| 1. Project Bootstrap and Data Layer | 1/1 | Complete ✓ |
| 2. Signal Matrix Generation | 1/1 | Complete ✓ |
| 3. Vectorized Backtesting Engine | 1/1 | Complete ✓ |
| 4. Combo Generation and Backtesting | 1/1 | Complete ✓ |
| 5. Validation Framework | 1/1 | Complete ✓ |
| 5b. Tail Analysis Automation | 1/1 | Complete ✓ |
| 6. Monte Carlo Integration | 1/1 | Complete ✓ |
| 7. RSI Search Entry Point | 1/1 | Complete ✓ |
| 8. Composer Export | 1/1 | Complete ✓ |
| 9. Output and Report Generation | 1/1 | Complete ✓ |

---
*Roadmap created: 2026-06-10*
