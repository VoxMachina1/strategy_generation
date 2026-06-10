# Codebase Synthesis — Cross-Cutting Findings

Generated: 2026-06-08 | Updated: 2026-06-09  
Source: Deep inventory pass across all local Python projects (excluding crescendo)

**See also:** `signal_pipeline/VISION.md` (authoritative architecture vision, written 2026-06-09) and updated `signal_pipeline/PLANNING.md` (signal type architecture, crisis hold-out filter, regime analysis, Mode C assembly design).

See sibling files for exhaustive per-tool inventories:
- `composer_signal_generator.md`
- `monte_carlo_sim.md`
- `rsi_tester.md`
- `strategy_viewer.md`
- `main_vs_main2_diff.md` (diff result)

---

## The Four Tools and Their True Roles

| Tool | Core Job | Data Source | Indicator Types | Output |
|---|---|---|---|---|
| `composer_signal_generator` | Signal discovery + walk-forward evaluation | yfinance | RSI, SMA, EMA, CumRet, STD, RETURNS_MA | CSVs + optional Composer DSL |
| `monte_carlo_sim` | Forward risk simulation of live Composer symphonies | Composer API + yfinance | N/A (consumes backtest output) | PNGs + CSVs |
| `rsi_tester` | RSI frontrunner discovery for a specific Composer strategy | Tiingo | RSI only | filtered.csv + strategy_modified.json |
| `strategy_viewer` | Parameter fragility + tail event analysis | Tiingo | RSI, SMA, EMA, CumRet, MaxDD, MAReturn | Self-contained HTML report |

---

## Data Source Inconsistency (Critical)

**composer_signal_generator and monte_carlo_sim use yfinance.**  
**rsi_tester and strategy_viewer use Tiingo.**

This means a signal discovered in composer_signal_generator and then validated in strategy_viewer is potentially being evaluated on different price data. Tiingo uses adjusted close prices from their own adjustment pipeline; yfinance uses Yahoo Finance's adjusted close. These diverge on specific dividend events and split dates.

**Target state:** All tools should use Tiingo. The `data_loader.py` + `data_alignment.py` modules already exist and are tested.

**Constraint:** True ATR computation requires OHLC data. The current Tiingo data layer downloads only adjusted close (`close` column). If ATR is added to the indicators library, either: (a) the Tiingo downloader must be extended to also fetch high/low, or (b) ATR must remain yfinance-only until the data layer is upgraded.

---

## Known Bugs (Confirmed)

### 1. Latent NameError in composer_signal_generator — frozen combo universe
- **File:** `main.py` and likely `main2.py`, lines ~3084–3141
- **Condition:** `freeze_combo_universe=True` in the config
- **What happens:** Code calls `summarize_combo_distributions()`, `build_combo_corr()`, `portfolio_from_shortlist()`, and `per_fold_ssopt()` — none of which are defined in either file. These were apparently in an external module that was deleted when the "single-file solution" was written. The flag `COMBO_MODULES_AVAILABLE = True` (line 44) means the guard does NOT protect against this.
- **Effect:** NameError crash when frozen combo universe is enabled
- **Fix:** Either port the four missing functions or remove the dead block

### 2. Monte Carlo drawdown unit mismatch
- **File:** `monte_carlo_sim/Monte Carlo walk forward composer working.py`
- **Location:** `plot_drawdown_distributions()` call site
- **What happens:** `run_monte_carlo_simulation()` stores `max_drawdowns` as fractions (0.15 = 15%). `analyze_drawdowns()` returns `max_drawdown` as a percent (15.0). Both are passed to `plot_drawdown_distributions()` which overlays them on the same histogram — one x-axis is fractions, the other is percent. The chart is visually wrong.
- **Fix:** Convert simulation max_drawdowns to percent before passing to the comparison chart (multiply by 100)

### 3. merge conflict in fuzz_tester.py — RESOLVED, cosmetic artifact only
- **File:** `strategy_viewer/fuzz_tester/fuzz_tester.py`, lines 497–503
- **Status:** The PLANNING.md and TASKS.md description of a `primary_returns` vs `all_returns` variable-name conflict is **incorrect for the current code**. The function body is correct and uses `all_returns` throughout. The only artifact is a 3-line comment block that appears twice consecutively:
  ```
  # ---------------------------------------------------------------------------
  # Single condition sweep
  # ---------------------------------------------------------------------------
  ```
- **Fix:** Delete the duplicate 3 lines (lines 501–503). No behavioral change.
- **TASKS.md Task 1.1 must be updated** to reflect this — the task as written is wrong.

### 4. Total_Trades bug in rsi_tester metrics
- **File:** `rsi_tester/strategy_engine/src/metrics.py`
- **Location:** `calculate_metrics()`, `Total_Trades` computation
- **What happens:** `total_trades = len(active_days)` counts active days, not trade entries. A 10-consecutive-day active streak counts as 10 trades, not 1.
- **Fix:** Count signal state transitions (0→1), not total active days. The `calculate_strategy_returns` function already has this logic (trade multiplier for state changes).

---

## Misleading Names (Confirmed)

### `_ATR` in composer_signal_generator
- **File:** `main.py` / `main2.py`, `_ATR()` function (~line 111)
- **What it actually does:** Rolling standard deviation of price (`price.rolling(n).std()`)
- **What ATR means:** Average True Range — requires high, low, and close to compute `max(H-L, H-prev_C, prev_C-L)` average
- **Constraint:** True ATR requires OHLC data. Current Tiingo data layer only downloads adjusted close.

### `_BBANDS` in composer_signal_generator
- **File:** `main.py` / `main2.py`, `_BBANDS()` function (~line 104)
- **What it actually does:** Returns only the upper Bollinger Band (`SMA + std_dev * std_multiplier`)
- **Fix needed:** Add `_BBANDS_lower()` returning the lower band, or return a tuple and rename to `_BBAND_UPPER()`

---

## Feature Inventory (What Exists Where)

### Walk-forward evaluation
- **Full implementation:** composer_signal_generator (Holdout + Walk-Forward + Expanding + Rolling, checkpointed, parallelized)
- **Not in:** rsi_tester (single full-period backtest only), strategy_viewer (no OOS eval), monte_carlo_sim (bootstrap simulation, not proper WF)

### Portfolio construction
- **Only in:** composer_signal_generator — ERC (SLSQP), Smart-Sharpe (cross-validated SLSQP multi-start), InvVol, low-correlation greedy shortlist
- **Not in:** rsi_tester (selects best individual signals), strategy_viewer (no portfolio), monte_carlo_sim (equal-weight only)

### Tail event analysis
- **Only in:** strategy_viewer — `compute_tail_metrics()`: tail_concentration, excess_kurtosis, stripped_win_rate, wr_delta, tail_score
- **Not in:** composer_signal_generator (no tail metrics), rsi_tester (no tail metrics), monte_carlo_sim (no tail metrics)

### Precondition support
- **composer_signal_generator:** AST-based sandboxed parser (`_safe_eval_precond`) — more secure, handles complex expressions, supports PRICE/SMA/EMA/RSI/BBANDS/ATR/ZSCORE
- **rsi_tester:** `df.eval()` — simpler, limited to pandas-eval-compatible column references
- **strategy_viewer:** No preconditions — analyzes conditions as found in the JSON

### Composer JSON integration
- **Read (parse tree):** rsi_tester (`strategy_paths.py`), strategy_viewer (`extract_conditions_from_tree()`)
- **Write (insert signals):** rsi_tester (`strategy_inserter.py`)
- **Read (via Composer API):** monte_carlo_sim (`fetch_backtest()`)

### Resume / checkpoint
- **Only in:** composer_signal_generator — master checkpoint + per-method iteration checkpoints

### Blackout date ranges
- **Only in:** composer_signal_generator

### Beat-rates comparison
- **Only in:** strategy_viewer — for each sweep point, tracks what fraction of signal days the endpoint beats every other loaded ticker

---

## Entry Point Map

| Entry point | Command | Prompts | Approximate runtime |
|---|---|---|---|
| `composer_signal_generator/main.py` | `python main.py` | 19 interactive | Hours |
| `composer_signal_generator/main2.py` | `python main2.py` | 19 interactive | Hours (max_workers differs) |
| `composer_signal_generator/analysis_workshop.py` | `python analysis_workshop.py` | 10 interactive | Seconds |
| `monte_carlo_sim/Monte Carlo walk forward composer working.py` | `python "Monte Carlo..."` | 3 interactive | Minutes |
| `rsi_tester/run_analysis.py` | `python run_analysis.py` | 1 interactive | Hours |
| `rsi_tester/strategy_engine/main.py` | `python strategy_engine/main.py` | None (config-driven) | Hours |
| `rsi_tester/pathfinder/strategy_paths.py` | `python pathfinder/strategy_paths.py` | None | Seconds |
| `rsi_tester/strategy_filter/filter_results.py` | `python strategy_filter/filter_results.py` | None | Seconds |
| `rsi_tester/strategy_inserter/strategy_inserter.py` | `python strategy_inserter/strategy_inserter.py [args]` | None | Seconds |
| `strategy_viewer/fuzz_tester/fuzz_tester.py` | `python fuzz_tester.py [json_path]` | 10 interactive | Minutes |

---

## rsi_tester Clarification (What It Actually Does)

This was the tool most easily forgotten. Its purpose: **discover RSI "frontrunner" signals for an existing Composer strategy**.

The workflow is strategy-centric, not signal-centric:
1. You give it a Composer strategy JSON
2. It finds every leaf asset (every path through the if/else tree)
3. For each leaf: it tests whether any RSI signal (across 15 hardcoded signal tickers, both overbought and oversold thresholds, 50→99 and 1→50) can identify days when holding a different target beats the current leaf asset
4. Filter: win_rate > 75%, total_trades > 20, benchmark_median_return < 0 (signal fires when the current asset is hurting)
5. Insert winners back as `wt-cash-equal` parallel votes in the strategy JSON

The key filter criterion — **benchmark_median_return < 0** — is the conceptual core. It ensures discovered signals are complementary to the existing strategy (they fire when the strategy's own assets are underperforming), not redundant with it.

---

## VISION.md Status

**Created 2026-06-09** at `signal_pipeline/VISION.md`. Covers both projects. Authoritative reference for: two signal types (Type 1 replacement, Type 2 regime), five-stage pipeline, three integration modes (Mode C primary), crisis hold-out filter design, stripped Sharpe, benchmark_median_return constraint, and key design decisions.

The `.planning/ROADMAP.md` in `strategy_viewer/` is an older GSD-framework roadmap covering only strategy_viewer phases — distinct from the new VISION.md.

---

## Suggested Unification Architecture (Opinion)

See the discussion in the session notes for the full recommendation. Summary:

The rsi_tester has the right conceptual shape (triplet model, strategy-centric discovery). The composer_signal_generator has the right evaluation depth (multi-indicator, walk-forward, portfolio construction). The strategy_viewer has the right quality analysis (tail events, fragility). These three should be phases of one pipeline, not separate tools.

The missing piece in all three: tail event analysis is currently only in the strategy_viewer and is manual/interactive. It should be automated and run as part of the discovery pipeline so that signals are ranked by a combined (OOS performance + tail health + fragility) score from the start.
