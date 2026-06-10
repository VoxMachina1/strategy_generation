# Inventory: `composer_signal_generator` Module

Generated: 2026-06-08

---

## Overview

This module is a self-contained quantitative trading signal discovery engine. Its purpose is to generate thousands of boolean trading signals from technical indicators (RSI, SMA, EMA, cumulative returns, standard deviation), then evaluate those signals rigorously using multiple out-of-sample testing methodologies: Holdout (70/30 with embargo), Walk-Forward, Expanding Window, and Rolling Window analysis.

The core workflow:
1. User specifies target tickers (assets to trade), reference tickers (indicators), and signal types interactively.
2. Price data is downloaded via yfinance (adjusted closes, max history, in parallel).
3. Thousands of boolean signals are generated (e.g. `RSI_14_SPY_GT_60`, `SMA_50_QQQ_GT_SMA_200_QQQ`).
4. Each signal is backtested in parallel (ProcessPoolExecutor) across each test window.
5. Synergistic AND/OR/gated combinations of the top signals are searched (again in parallel).
6. Aggregate statistics, distribution summaries, and a low-correlation portfolio shortlist are written to a structured folder hierarchy.
7. A secondary analysis script (`analysis_workshop.py`) allows the user to post-process and interactively filter the output CSV files.

The module is designed specifically for the [Composer.trade](https://composer.trade) algorithmic trading platform, and can optionally convert discovered signals into Composer DSL code via `composer-tools`.

---

## Entry Points

### `main.py` — primary engine
**Run with:** `python main.py`

**Interactive prompts (in order):**
1. Run name (used for folder naming and resume detection)
2. Target tickers (comma-separated; e.g. `TECL,QQQ,BIL`)
3. Reference/indicator tickers (e.g. `KMLM,CORP,XLU,XLK,SPY`)
4. Benchmark ticker (default `SPY`)
5. Safe asset for Composer "else" condition (default `BIL`)
6. Signal types: RSI / CUMRET / RETURNS_MA / STD / PRICE_SMA / PRICE_EMA / ALL / CUSTOM
7. RSI period configuration (default vs. custom period pairs)
8. SMA/EMA/RETURNS_MA period lists (if those signal types selected)
9. Sorting metric (Smart Sharpe / Sharpe / Sortino / Calmar / Total Return / Robustness)
10. Time-in-Market minimum, Max Drawdown maximum, Quantile filter cutoff
11. Evaluation methods: Holdout / Walk-Forward / Expanding / Rolling / All
12. Window parameters (train days, test days, step size, embargo days) per method
13. Synergistic combos: enabled/disabled, K_primary, M_partner, min_train_gain, min_test_gain, max_legs
14. Frozen combo universe: enabled/disabled, universe_size, source (holdout_train vs. first_wf_train)
15. Correlation analysis and portfolio construction: enabled/disabled, min_overlap, portfolio method (invvol/erc)
16. Smart-Sharpe portfolio optimization: enabled/disabled, k_folds, w_cap, n_starts
17. Combo export settings: frozen_top_k, corr_threshold, shortlist_size
18. Signal preconditions (e.g. `PRICE('SPY') > SMA('SPY',200)`)
19. Blackout date ranges (e.g. `2020-03-20 to 2020-12-31`)

**Produces at the end:**
- A folder tree under `datasets/<run_name>/run_<N>_<method>/`:
  - `aggregates/evaluation_summary.csv` — top-50 leaderboard per method
  - `aggregates/method_averages.csv` — per-signal distribution stats across all iterations
  - `combos/combo_quant_summary_frozen.csv` — distribution stats for frozen combo universe
  - `combos/combo_lowcorr_shortlist_frozen.csv` — greedy low-correlation portfolio of combos
  - `combos/portfolio_series_equal_weight.csv` — daily equity curve of shortlisted portfolio
  - `combos/portfolio_weights_erc_frozen.csv` — ERC portfolio weights
  - `combos/portfolio_weights_smartsharpe_frozen.csv` — Smart-Sharpe optimizer weights
  - `holdout/results.csv`, `walk_forward/results.csv`, `expanding/results.csv`, `rolling/results.csv`
  - Per-iteration combo logs in `walk_forward/iters/iteration_combo_logs/`
  - `README.md` (auto-generated analysis guide), `manifest.csv` (index of all files)
  - `cache/prices_<hash>.pkl`, `cache/signals_<hash>.pkl` — disk caches for resume
  - `master_checkpoint.pkl` — serialized run state for crash recovery
- Optionally: `composer_signals.txt` — Composer-ready DSL code for top signals

**Supports resume:** if `master_checkpoint.pkl` exists for the named run, the script resumes from the last completed evaluation method. Each method (walk-forward, rolling, expanding) also has its own iteration-level `checkpoint.pkl`.

### `main2.py` — near-identical variant
**Run with:** `python main2.py`

Functionally identical to `main.py`. The only confirmed difference found is `ProcessPoolExecutor(max_workers=6)` in `enrich_with_synergistic_combos` vs `max_workers=5` in `main.py`. The file is 5524 lines vs. 5070 lines; the additional ~450 lines appear to be expanded/duplicated sections. This appears to be a development branch or snapshot that was never cleaned up.

### `analysis_workshop.py` — post-processing filter tool
**Run with:** `python analysis_workshop.py`

**Reads from:** `analysis_inbox/` folder — all `*.csv` files are loaded and merged.

**Interactive prompts (in order):**
1. Minimum `HitRate_Positive_Sharpe` (default 0.75)
2. Minimum `N_Iterations` (default 1)
3. Minimum `Sharpe_p10` worst-case Sharpe (default 0.5)
4. Minimum `Sharpe_p50` median Sharpe (default 1.5)
5. Minimum `Robustness_Score_mean` (default 0.55)
6. Maximum `Sharpe_CoV` stability filter (default 1.5)
7. Minimum `Median_Calmar` drawdown efficiency (default 2.0)
8. Maximum `Sharpe_p90` anti-home-run filter (default 5.0)
9. Optional: additional custom filters on any numeric column
10. Final sort column and direction

**Produces:**
- `analysis_output/filtered_results_<YYYYMMDD_HHMMSS>.csv`

---

## Functions & Classes (Exhaustive)

### `main.py` (and `main2.py` — identical unless noted)

#### Preconditions Engine (lines ~49–185)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_PRICE` | ~85 | `prices: DataFrame, tkr: str` | Extracts a single ticker's price series from the price panel. Falls back to NaN series if ticker missing. | `pd.Series` | None |
| `_SMA` | ~94 | `prices: DataFrame, tkr: str, n: int` | Rolling simple moving average of `_PRICE(tkr)` over `n` periods. | `pd.Series` | None |
| `_EMA` | ~97 | `prices: DataFrame, tkr: str, n: int` | Exponential moving average (ewm, adjust=False). | `pd.Series` | None |
| `_RSI` | ~100 | `prices: DataFrame, tkr: str, n: int` | RSI via `ta.momentum.RSIIndicator`. | `pd.Series` | None |
| `_BBANDS` | ~104 | `prices: DataFrame, tkr: str, n: int, std: float=2.0` | Bollinger upper band (SMA + std_dev * std). Note: returns only upper band, not a full tuple. | `pd.Series` | None |
| `_ATR` | ~111 | `prices: DataFrame, tkr: str, n: int` | Average True Range approximation via rolling std of price (NOT true ATR — no high/low used). | `pd.Series` | None |
| `_ZSCORE` | ~117 | `prices: DataFrame, tkr: str, n: int` | Z-score of price relative to rolling mean/std. | `pd.Series` | None |
| `_safe_eval_precond` | ~143 | `expr: str, prices: DataFrame` | Safely evaluates a user-supplied expression string (e.g. `PRICE('SPY') > SMA('SPY',200)`) using AST whitelist. Raises `ValueError` on bad syntax or disallowed nodes. | `pd.Series` (bool) | None |
| `build_precondition_series` | ~168 | `prices: DataFrame, preconds: list[str], combine: str="AND"` | Combines multiple precondition expressions with AND or OR logic. Returns all-True series if `preconds` is empty. | `pd.Series` (bool) | None |

#### Progress / Download Utilities (lines ~189–511)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_tqdm` | ~191 | `x=None, **kwargs` | Stub that disables all tqdm progress bars; returns the iterable unchanged. | iterable or range | None |
| `PromptState` (class) | ~203 | — | Caches user prompt answers so repeated prompts return the cached answer without re-asking. Methods: `ask_bool_once`, `set`, `get`, `get_smart_sharpe`. | — | — |
| `PM` | ~227 | — | Module-level singleton `PromptState` instance. | — | — |
| `_download_prices_adj_with_bar` | ~229 | `tickers, start, end, desc` | Per-ticker yfinance download of Adj Close with tqdm bar. Falls back to Close. | `pd.DataFrame` | Network I/O |
| `_download_prices` | ~248 | `tickers, start, end, period, desc` | Single or multi-ticker yfinance download. Prefers Adj Close. For multiple tickers, downloads one at a time with progress bar. | `pd.DataFrame` | Network I/O |
| `_download_single_ticker_safe` | ~296 | `ticker_symbol: str` | Single ticker download with 3-attempt retry (0s, 2s, 5s delay). Uses `yf.Ticker().history` as fallback. Returns None on total failure. | `pd.DataFrame` or `None` | Network I/O, prints on failure |
| `download_prices_max_debug` | ~341 | `tickers` | Parallel download of max history for all tickers using `ThreadPoolExecutor(max_workers=10)`. Deduplicates columns. | `pd.DataFrame` | Network I/O, prints progress |
| `availability_report` | ~385 | `px` | Prints first/last date and row count for each ticker. | `pd.DataFrame` (report) | Prints |
| `quick_smoke_test` | ~404 | — | Tests downloads for a hardcoded set of 12 tickers. Returns price DataFrame or None on failure. | `pd.DataFrame` or `None` | Network I/O, prints |
| `debug_namespace_pollution` | ~421 | — | Checks for global namespace pollution by common variable names (`t`, `df`, etc.). | None | Prints |
| `trim_to_overlap` | ~456 | `px, min_overlap=60` | Trims price panel to rows where all tickers have data. Falls back to soft overlap (dropping tickers with insufficient data) if strict overlap is too short. | `pd.DataFrame` | May print warning |

#### Combo Analysis Superset (lines ~472–994)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_is_combo_row` | ~479 | `row` | Detects if a DataFrame row represents a combo signal (by Combo_Op column or presence of `+AND+`/`+OR+` in Signal name). | `bool` | None |
| `_parse_combo_recipe` | ~485 | `signal_name` | Parses combo signal name like `A+AND+B+OR+C` into `(members, ops)` lists. | `(list, list)` or `(None, None)` | None |
| `_combine_op` | ~512 | `a, b, op` | Applies AND/OR/A_AND_NOT_B/B_AND_NOT_A boolean operation on two Series. | `pd.Series` (bool) | None |
| `_combine_recipe` | ~519 | `members, ops, signals` | Chains multiple signals using a sequence of ops from the signals dict. | `pd.Series` (bool) | None |
| `_parse_period_str` | ~526 | `s` | Parses "YYYY-MM-DD to YYYY-MM-DD" string into two Timestamps. | `(Timestamp, Timestamp)` | None |
| `_slice_prices` | ~532 | `price_df, start, end` | Slices price DataFrame to a date range. | `pd.DataFrame` | None |
| `_bt_signal` | ~537 | `sig_series, price_slice, ticker, precond_mask=None` | Backtests a boolean signal series on a price slice for a given ticker. Returns quantstats metrics dict plus Time_in_Market and Signal_Returns. | `dict` | None |
| `_enumerate_windows` | ~551 | `all_results` | Enumerates all test windows across all methods (holdout, walk-forward, expanding, rolling). | `list[dict]` | None |
| `_seed_combo_frame` | ~570 | `all_results` | Selects the best available result set (preferring Holdout filtered_results) to use as the seed for freezing the combo universe. | `(DataFrame, str)` | None |
| `_freeze_combos` | ~589 | `all_results, top_k=30` | Picks the top `top_k` combos from the seed window by Smart Sharpe + Total Return. Returns a list of parsed combo dicts. | `list[dict]` | None |
| `_eval_frozen_combos` | ~609 | `frozen, windows, signals, full_prices` | Evaluates the frozen combo set across all enumerated windows. | `pd.DataFrame` | None |
| `_gather_dynamic_combo_oos` | ~639 | `all_results` | Collects combo OOS metrics from all methods into a single long panel. Handles missing/None results defensively. | `pd.DataFrame` | None |
| `_dist_summary` | ~700 | `oos_df` | Aggregates per-combo stats across windows: N_Iterations, Sharpe_p50/p10/p90/IQR, Return_p50/p10/p90, MaxDD_p90, HitRate_Positive_Sharpe. | `pd.DataFrame` | None |
| `_lowcorr_shortlist` | ~737 | `oos_df, summary_df, metric, corr_threshold=0.30, max_keep=12` | Greedy forward-selection of low-correlation combos by Spearman rank correlation across test periods. | `pd.DataFrame` | None |
| `_erc_weights` | ~769 | `oos_df, names, metric` | Simple iterative ERC (Equal Risk Contribution) weight solver using rank covariance. | `pd.Series` (weights) | None |
| `_smart_sharpe_opt` | ~787 | `oos_df, names, ret_metric, risk_metric` | Random coordinate-search optimizer maximizing median(ret)/p90(|drawdown|). 2000 iterations. | `pd.Series` (weights) | None |
| `_invvol_weights_robust` | ~812 | `rets: DataFrame, cap: float=None` | Inverse-volatility weights with optional cap and normalization. | `pd.Series` | None |
| `_erc_weights_robust` | ~829 | `rets: DataFrame, cap: float=None` | ERC via SLSQP optimization with fallback to invvol. | `pd.Series` | None |
| `_smart_sharpe_opt_robust` | ~865 | `rets: DataFrame, cv_folds=3, cap=0.35, starts=12, method="erc", rng_seed=123` | Cross-validated weight search maximizing mean OOF Smart Sharpe. Falls back to base method. | `pd.Series` | None |
| `export_all_combo_artifacts` | ~906 | `all_results, signals, full_prices, output_dir, name_prefix, frozen_top=30, corr_threshold=0.30, shortlist_size=12, write_weights=True` | Master exporter. Writes dynamic and frozen combo CSVs and optional ERC/Smart-Sharpe weight files to `combos/` subdirectory. | None | Writes ~11 CSV files |

#### Blackout Ranges (lines ~996–1119)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `has_plus` | ~999 | `series` | Fast vectorized check if strings in a Series contain `+`. | `pd.Series` (bool) | None |
| `_parse_blackout_chunks` | ~1006 | `user_text: str` | Parses user-supplied blackout range strings (various separators) into list of `(start, end)` string pairs. | `list[tuple]` | None |
| `get_blackout_ranges_from_user` | ~1029 | — | Prompts user for blackout ranges. Accepts default, 'none', or custom. | `list[tuple]` | Prints, reads stdin |
| `get_preconditions_from_user` | ~1064 | — | Prompts user for precondition expression strings and combine mode. | `(list[str], str)` | Prints, reads stdin |
| `apply_blackout_ranges` | ~1104 | `price_df: DataFrame, ranges` | Sets price values to NaN for each blackout window in the price panel. | `pd.DataFrame` | None |

#### Portfolio Series Builder (lines ~1121–1327)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_is_combo_name` | ~1126 | `_s` | String check for combo-style signal names. | `bool` | None |
| `_pick_method_df` | ~1132 | `all_results, method_name` | Returns the appropriate DataFrame for a given method name from all_results. | `pd.DataFrame` or `None` | None |
| `_extract_combo_returns_panel` | ~1146 | `all_results, signals_set, method_name` | Builds a wide (dates × signals) panel of daily returns from the chosen method's OOS test windows. | `pd.DataFrame` or `None` | None |
| `_load_shortlist_signals` | ~1183 | `shortlist_csv_path` | Reads Signal column from a shortlist CSV. | `list[str]` | Reads file |
| `_load_ssopt_weights` | ~1191 | `weights_csv_path` | Reads weight CSV. Handles two formats: (Signal, Weight) table or (index, weight) series CSV. | `dict` | Reads file |
| `_compute_portfolio_series` | ~1217 | `panel, weights, renormalize_daily=True` | Computes daily portfolio return series from a returns panel and a weight dict. With `renormalize_daily=True`, re-normalizes weights each day over signals that have non-null data. | `pd.Series` or `None` | None |
| `build_and_write_portfolio_series` | ~1264 | `all_results, output_dir, shortlist_csv, weights_ssopt_csv=None, method_preference=(...), renormalize_daily=True` | Builds and writes `portfolio_series_equal_weight.csv` and optionally `portfolio_series_ssopt.csv`. | None | Writes 1–2 CSV files |

#### Execution Mode & Helpers (lines ~1329–1479)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `align_signal_and_returns` | ~1337 | `signal, returns` | Shifts signal/returns based on `EXECUTION_MODE` global (`MOC` = Composer-like: signal@t × return@t+1; `NEXT_BAR` = classic). | `(signal, returns)` | None |
| `_cleanup_tqdm` | ~1346 | — | Closes any open tqdm progress bars. | None | None |
| `safe_print` | ~1366 | `*args, **kwargs` | Print with `flush=True`. | None | Prints |
| `safe_input` | ~1370 | `prompt, default=""` | Input that flushes stdout, handles EOFError gracefully. | `str` | Reads stdin |
| `_parse_periods` | ~1381 | `prompt_text, default_list` | Parses comma-separated integers from user input with validation. | `list[int]` | Reads stdin |
| `normalize_price_panel` | ~1395 | `df` | Normalizes price panel: resolves MultiIndex columns to single price field, deduplicates columns, ensures numeric dtype. | `pd.DataFrame` | None |
| `_series` | ~1418 | `df, t` | Extracts a 1-D float Series for ticker `t` from the price panel. Handles duplicate column case. | `pd.Series` | None |
| `unique` | ~1426 | `seq` | Order-preserving deduplication. | `list` | None |
| `_fmt_date` | ~1436 | `d` | Robust date-to-string formatter. | `str` | None |
| `report_blackout_status` | ~1444 | `df, label` | Prints blackout block summary for a price panel. | `(pd.Series, np.array)` | Prints |
| `_yf_download_with_retry` | ~1462 | `tickers, **kwargs` | yfinance download with 3-attempt retry. | `pd.DataFrame` | Network I/O |
| `_yf_download_with_retry_adj_with_bar` | ~1480 | `tickers, start, end, period, desc` | Per-ticker download with retry and tqdm bar. | `pd.DataFrame` | Network I/O |

#### Combo Distribution & File Saving (lines ~1513–2044)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_is_combo_row` (duplicate) | ~1513 | `row` | Second definition; detects combo rows (uses `+AND+`/`+OR+` etc.). Slightly different from the one in the superset. | `bool` | None |
| `_percentile` | ~1521 | `a, q` | Safe `np.percentile` for a pandas-coercible series. | `float` | None |
| `_iqr` | ~1527 | `a` | IQR (75th–25th percentile). | `float` | None |
| `_hit_rate_pos` | ~1533 | `x` | Fraction of values > 0. | `float` | None |
| `_build_combo_distribution` | ~1539 | `df_method: DataFrame, method_label: str` | Builds distributional statistics (p10/p50/p90/IQR/std/hit_rate) for combos within a single method's results. | `pd.DataFrame` | None |
| `RunPaths` (dataclass) | ~1603 | `root: str` | Dataclass that auto-creates and stores paths for all output subdirectories (aggregates, holdout, walk_forward, expanding, rolling). Methods: `method_dir`, `iter_dir`. | — | Creates directories |
| `save_df` | ~1633 | `df, folder, filename, kind, method, note, prefix` | Saves DataFrame to CSV (dropping `Signal Returns`). Also saves a combo-only version if combos present. Appends to `manifest_rows`. | `str` (path) | Writes CSV files |
| `write_readme` | ~1663 | `paths: RunPaths, cfg_summary: str` | Generates and writes a `README.md` with analysis workflow guidance and configuration summary. | None | Writes file |
| `write_manifest` | ~1719 | `paths: RunPaths` | Writes `manifest.csv` from the accumulated `manifest_rows` global list. | None | Writes file |
| `_write_combo_distribution_files` | ~1724 | `all_results: dict, paths, name_prefix: str, shortlist_filters: dict=None` | Builds per-method and combined combo distribution files. Applies optional quant-style gate filters. Writes many CSV files including basic stats, percentiles, hit rates, stability, and coverage sub-files. This function has deeply nested conditional logic that generates many redundant files. | `(str, str)` | Writes many CSV files |
| `_drop_heavy_cols` | ~3180 | `df` | Drops `Signal Returns` column from a DataFrame. | `pd.DataFrame` | None |

#### Evaluation Classes & Config (lines ~2046–2081)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `EvalMode` (Enum) | ~2046 | — | Enum with values: `HOLDOUT_70_30`, `WALK_FORWARD`, `EXPANDING`, `ROLLING`. | — | — |
| `EvaluationConfig` | ~2052 | — | Class holding all window parameters with defaults: holdout 70/30, embargo 5 days, WF train 252/test 63/step 21 days, expanding initial 252/test 63/expansion 63, rolling train 252/test 63/step 21, MC 10000 sims, robustness cutoff 0.5. | — | — |

#### Data Split & Monte Carlo (lines ~2082–2249)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `calculate_embargo_split` | ~2082 | `price_data, train_pct, embargo_days=0, valid_on=None` | Splits on valid (non-NaN) bars rather than raw row count. Returns train_data, test_data, and embargo indices. | `(DataFrame, DataFrame, int, int)` | Prints split summary |
| `run_monte_carlo_validation` | ~2149 | `returns, num_simulations=10000, simulation_length=None, annual_periods=252, random_state=None` | Bootstrap Monte Carlo: draws random returns preserving empirical positive/negative rate. Returns paths, final returns, MDD distribution, and percentile bands. | `dict` or `None` | None |
| `evaluate_signal_performance` | ~2201 | `signal_returns, benchmark_returns=None, mode="OOS", period_name="", config=None` | Computes quantstats metrics plus Monte Carlo validation (MC percentile, coverage, expected MDD). | `dict` or `None` | None |

#### Walk-Forward / Rolling / Expanding Evaluators (lines ~2252–2734)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `run_walk_forward_evaluation` | ~2252 | `signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None` | Runs walk-forward analysis: non-overlapping train/test windows stepped by `wf_step_size`. Checkpoints every 5 iterations. Runs combo enrichment per iteration if enabled. | `pd.DataFrame` | Writes CSV files, checkpoint PKL, prints ETA |
| `run_rolling_window_evaluation` | ~2415 | `signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None` | Fixed-size rolling window evaluation. Same checkpoint/ETA/combo logic as walk-forward. | `pd.DataFrame` | Writes CSV files, checkpoint PKL, prints ETA |
| `run_expanding_window_evaluation` | ~2575 | `signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None` | Expanding window evaluation (training set grows each iteration). Same checkpoint/ETA/combo logic. | `pd.DataFrame` | Writes CSV files, checkpoint PKL, prints ETA |
| `run_comprehensive_evaluation` | ~2737 | `signals, price_data, target_tickers, eval_modes, config, paths, name_prefix="", base_cfg=None, completed_evaluations=None` | Orchestrator: runs all selected methods in sequence, applies preconditions to signals first, runs final reporting at end. Resume-aware (skips already-completed methods). | `(dict, list)` | Writes many files, prints progress |

#### Smart-Sharpe Portfolio Optimizer (lines ~2874–3081)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_smart_sharpe_safe` | ~2881 | `series` | Computes `qs.stats.smart_sharpe` safely; returns -inf on failure. | `float` | None |
| `_split_contiguous_folds` | ~2893 | `index, k` | Splits an integer range into k contiguous folds. | `list[np.array]` | None |
| `optimize_weights_smart_sharpe` | ~2898 | `R: DataFrame, k_folds=1, w_cap=0.35, n_starts=8, random_state=42` | SLSQP optimizer maximizing Smart Sharpe (or worst-fold Smart Sharpe if k_folds>1) subject to sum=1, 0≤w≤cap. Multi-start with equal/cap-spread/Dirichlet initializations. | `(np.array, pd.Series)` | Prints metrics |
| `build_portfolio_smart_sharpe` | ~3005 | `all_results, shortlist_csv_path, weight_scheme="equal"` | Collects OOS daily returns for shortlisted combos from all_results, builds equal-weight portfolio, computes Smart Sharpe/Sharpe/Sortino. | `(pd.Series, pd.DataFrame, np.array)` or `(None, None, None)` | Prints metrics |

#### Reporting & Filtering (lines ~3174–3496)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `fmt_pct` | ~3174 | `x` | Formats a decimal as a percentage string (e.g. 0.123 → "12.30%"). | `str` | None |
| `save_method_averages` | ~3184 | `all_results, paths, name_prefix="", per_method_files=False` | Computes per-signal aggregates (mean/median/std/min/max/CoV/IQR/percentiles/hit_rates) across all iterations for WF/expanding/rolling. Writes `method_averages.csv` and per-method `averages.csv`. | None | Writes CSV files |
| `generate_evaluation_summary` | ~3332 | `all_results, paths, name_prefix="", top_n_per_method=50` | Generates `evaluation_summary.csv` with top-N signals per method. Includes risk warnings based on holdout max drawdown. | None | Writes CSV file, prints |

#### User Input Collection (lines ~3497–3789)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `get_enhanced_user_inputs` | ~3497 | — | Full interactive configuration gathering: runs `get_user_inputs()` then layers on eval methods, combo, portfolio, and precondition settings. | `dict` (base_cfg) | Reads stdin, prints |
| `get_eval_only` | ~3740 | — | Asks only for evaluation method selection and window parameters (for re-running evaluations without regenerating signals). | `(list[EvalMode], EvaluationConfig)` | Reads stdin, prints |
| `get_user_inputs` | ~4573 | — | Core configuration: target tickers, reference tickers, benchmark, safe asset, signal types, RSI/SMA/EMA periods, sorting metric, TiM/MDD/quantile filters. | `dict` | Reads stdin, prints |

#### Signal Generation (lines ~3791–3956)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `generate_signals` | ~3792 | `tickers, price_data, signal_types, rsi_periods=None, price_sma_periods=None, price_ema_periods=None, returns_ma_periods=None` | Core signal factory. Generates thousands of boolean signals for each signal type: RSI (level thresholds + cross-period comparisons), CUMRET (level thresholds + cross-ticker/period), RETURNS_MA (cross-ticker/period comparisons), STD (cross-ticker/period), PRICE_SMA (price vs SMA, SMA vs SMA), PRICE_EMA (price vs EMA, EMA vs EMA). Stores all as boolean pd.Series in a dict. | `dict[str, pd.Series]` | Prints progress |

#### Main Entry Point (lines ~3960–4113)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `enhanced_main` | ~3960 | — | Top-level entry point. Handles resume detection from master checkpoint, fresh run setup, data caching, blackout application, and the per-method evaluation loop. Cleans up checkpoints on completion. | None | Full pipeline: downloads, generates signals, runs evaluations, writes all output, reads stdin |

#### Backtesting Core (lines ~4116–4571)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `convert_signal_to_composer_format` | ~4116 | `signal_name, target_ticker, safe_asset="BIL"` | Converts a signal name to Composer DSL code via `composer_tools`. Returns None if library not available. | `str` or `None` | None |
| `_sanitize_returns` | ~4136 | `r` | Converts any input to a float64 pd.Series with NaN/inf → 0.0. | `pd.Series` | None |
| `calculate_robustness_score` | ~4157 | `train_metric, test_metric` | Computes robustness as min(test/train, train/test). Returns 0 if either is ≤0. | `float` | None |
| `apply_comprehensive_filter` | ~4167 | `results, tim_min=0.025, mdd_max=-0.75, quant_filter=0.66, robustness_cutoff=0.0` | Applies four sequential filters: time-in-market, max drawdown, top-quantile, robustness score. | `pd.DataFrame` | Prints counts |
| `apply_robustness_filter` | ~4204 | `results, robustness_cutoff` | Legacy: filters by robustness cutoff only. | `pd.DataFrame` | None |
| `generate_composer_output` | ~4215 | `results, top_n=10, safe_asset="BIL", result_type="test", robustness_cutoff=0.0` | Generates Composer DSL code for top-N signals. | `list[dict]` or `None` | None |
| `save_composer_signals` | ~4254 | `composer_signals, filename` | Writes Composer signals to a formatted text file. | None | Writes file |
| `display_composer_preview` | ~4285 | `composer_signals, show_top=3` | Prints top N Composer signals to console. | None | Prints |
| `report_data_availability` | ~4312 | `close_prices: DataFrame, label: str` | Prints first valid date per ticker, overlap start date, and limiting tickers. | `(Timestamp, list)` | Prints |
| `get_initial_price_data` | ~4345 | `tickers: list` | Downloads max history, finds common start date, trims and drops NaN rows. | `pd.DataFrame` | Network I/O, prints |
| `calculate_quantstats_metrics` | ~4387 | `returns, benchmark_returns=None` | Computes comprehensive metrics: Total Return, Sharpe, Smart Sharpe, Sortino, Calmar (manual), Max Drawdown (manual), VaR, CVaR, Volatility, Skewness, Kurtosis, Win Rate, Best/Worst Day, Avg Win/Loss. Uses manual drawdown calculation to avoid quantstats Timedelta bug. | `dict` | None |
| `_run_single_backtest` | ~4460 | `args: tuple` | Worker function for parallel backtesting. Unpacks `(signal_name, signal, price_data, target_ticker, daily_ret, EXECUTION_MODE)`. Computes returns, metrics, and Time in Market. | `dict` | None |
| `backtest_signals` | ~4497 | `signals, price_data, target_tickers, benchmark_data=None, period_name=""` | Parallel backtesting using `ProcessPoolExecutor(max_workers=5)`. One task per (signal × ticker) combination. | `pd.DataFrame` | Process pool |
| `merge_train_test_results` | ~4521 | `train_results, test_results, sort_by` | Inner-joins train and test results on (Signal, Ticker), computes three robustness scores (Sharpe, Return, Sortino) and their average. Sorts by OOS performance and by robustness. | `(pd.DataFrame, pd.DataFrame)` | Prints |

#### Combo Generation (lines ~4754–5062)

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `_combine_series` | ~4754 | `a: Series, b: Series, op: str` | Applies AND/OR/A_AND_NOT_B/B_AND_NOT_A to two boolean series. | `pd.Series` | None |
| `_combine_many` | ~4762 | `series_list, ops_list` | Chains multiple series with a sequence of operations. | `pd.Series` | None |
| `_get_metric` | ~4769 | `df, name, col, default=-np.inf, ticker=None` | Safe metric lookup from a DataFrame/MultiIndex (handles single-ticker and cross-ticker modes). | `float` | None |
| `_greedy_build_combo` | ~4790 | `a_name, signals, partners, ops, train_data, test_data, tkr, tr_map, te_map, sort_by, max_legs, min_train_gain, min_test_gain, enable_cross_ticker, signal_ticker_map` | Greedy forward-selection for 3+ leg combos. Starts from `a_name`, iteratively adds the partner+op that most improves the sort metric above the gain threshold. Used when `max_legs > 2`. | `dict` or `None` | None |
| `_init_worker` | ~4881 | `signals_data, train_df, test_df, tr_map_data, te_map_data, sig_ticker_map_data` | ProcessPoolExecutor initializer: stores large data objects in worker-process global variables to avoid repeated serialization. | None | Sets process globals |
| `_find_combos_for_primary` | ~4897 | `args: tuple` | Worker: finds all synergistic partners for a given primary signal. For `max_legs <= 2`, exhaustive pairwise search; for `max_legs > 2`, delegates to `_greedy_build_combo`. Filters by train and test gain thresholds. | `list[dict]` | None |
| `enrich_with_synergistic_combos` | ~4997 | `signals, train_data, test_data, target_tickers, train_results, test_results, sort_by, K_primary=30, M_partner=40, ops=(...), min_train_gain=0.05, min_test_gain=0.00, random_state=42, show_progress=True, progress_leave=False, max_legs=2, enable_cross_ticker=True` | Master combo enrichment function. Selects top K_primary signals, searches up to M_partner random partners for each, runs in parallel via ProcessPoolExecutor with shared-memory initializer pattern. Returns all combos that pass gain thresholds, sorted descending by sort_by. | `pd.DataFrame` | Process pool, prints progress |

---

### `analysis_workshop.py`

| Name | Line | Parameters | What it does | Returns | Side effects |
|---|---|---|---|---|---|
| `safe_input` | ~14 | `prompt, default=None` | Input with default display; returns default if user presses Enter. | `str` | Reads stdin |
| `load_and_merge_data` | ~19 | `folder_path: Path` | Globs all `*.csv` in folder, loads each, concatenates into one DataFrame. | `pd.DataFrame` | Reads files, prints |
| `interactive_filter_menu` | ~47 | `df: DataFrame` | Guides user through 8 sequential filters (hit rate, N_iterations, Sharpe_p10, Sharpe_p50, robustness, CoV, Calmar, anti-home-run p90) plus optional custom filters on any numeric column. | `pd.DataFrame` | Reads stdin, prints |
| `main` | ~203 | — | Entry point: loads inbox CSVs, pre-calculates `Median_Calmar` (Return_p50 / abs(MaxDD_p90)), runs `interactive_filter_menu`, asks for sort column, writes timestamped output CSV. | None | Reads files, reads stdin, writes file |

---

## Data Structures

### `base_cfg` dict (returned by `get_enhanced_user_inputs`)
```
{
  'run_name': str,
  'target': list[str],          # tickers to trade
  'tickers': list[str],         # reference/indicator tickers
  'benchmark': str,
  'safe_asset': str,
  'signal_types': list[str],    # e.g. ['RSI', 'PRICE_SMA']
  'sort_by': str,               # e.g. 'Smart Sharpe'
  'tim': float,                 # time-in-market minimum
  'mdd': float,                 # max drawdown maximum
  'quant': float,               # quantile filter cutoff
  'rsi_periods': list[tuple] or None,
  'price_sma_periods': list[int] or None,
  'price_ema_periods': list[int] or None,
  'returns_ma_periods': list[int] or None,
  'eval_modes': list[EvalMode],
  'eval_config': EvaluationConfig,
  'enable_synergistic_combos': bool,
  'k_primary': int,
  'M_partner': int,
  'min_train_gain': float,
  'min_test_gain': float,
  'max_combo_legs': int,
  'freeze_combo_universe': bool,
  'combo_universe_size': int,
  'combo_universe_source': str,
  'combo_corr_min_overlap': int,
  'portfolio_method': str,      # 'invvol' or 'erc'
  'ssopt_enable': bool,
  'ssopt_cfg': dict,
  'frozen_top_k': int,
  'combo_corr_threshold': float,
  'combo_shortlist_size': int,
  'preconditions': list[str],
  'precondition_combine': str,  # 'AND' or 'OR'
}
```

### `all_results` dict (returned by `run_comprehensive_evaluation`)
```
{
  'holdout': {
    'merged_results': pd.DataFrame,   # all signals, pre-filter
    'robust_results': pd.DataFrame,   # sorted by Robustness_Score
    'filtered_results': pd.DataFrame, # post-filter (TiM, MDD, quantile, robustness)
    'train_period': str,              # "YYYY-MM-DD to YYYY-MM-DD"
    'test_period': str,
    'embargo_days': int,
  },
  'walk_forward': pd.DataFrame,   # all iterations combined
  'expanding': pd.DataFrame,
  'rolling': pd.DataFrame,
}
```

### Per-signal result row (columns in result DataFrames)
```
Signal, Ticker, Total Return, Sharpe Ratio, Smart Sharpe, Sortino Ratio,
Calmar Ratio, Max Drawdown, VaR (95%), CVaR (95%), Volatility, Skewness,
Kurtosis, Win Rate, Best Day, Worst Day, Avg Win, Avg Loss,
Time in Market, Signal Returns (pd.Series),
Train_Total_Return, Train_Smart_Sharpe, Train_Sharpe_Ratio,
Train_Sortino_Ratio, Train_Calmar_Ratio, Train_Max_Drawdown,
Robustness_Sharpe, Robustness_Return, Robustness_Sortino, Robustness_Score,
WF_Iteration / EW_Iteration / Roll_Iteration (method-specific),
Train_Period, Test_Period, Train_Days, Test_Days
```

### For combo rows, additional columns:
```
Combo_Op, Member_A, Member_B, Best_Member_Test, Best_Member_Train,
Synergy_Test, Synergy_Train, Member_A_Ticker, Member_B_Ticker, Is_Cross_Ticker
```

### `combo_quant_summary_frozen.csv` / `_dist_summary` output schema
```
Signal, Ticker, N_Iterations,
Sharpe_p50, Sharpe_p10, Sharpe_p90, Sharpe_IQR,
Return_p50, Return_p10, Return_p90,
MaxDD_p90, HitRate_Positive_Sharpe
```

### `method_averages.csv` schema (from `save_method_averages`)
```
Method, Signal, Ticker,
Total Return_mean/median/std/min/max,
Smart Sharpe_mean/median/std/min/max,
Sharpe Ratio_mean/median/std,
Sortino Ratio_mean/median/std,
Calmar Ratio_mean/median/std,
Max Drawdown_mean/median/std/min/max,
Time in Market_mean/median,
Robustness_Score_mean/median,
HitRate_Positive_Return, HitRate_Positive_Sharpe, N_Iterations,
Sharpe_CoV, Return_CoV,
Sharpe_p10/p25/p50/p75, Return_p10/p50, Return_gmean, MaxDD_p90,
Sharpe_IQR, RankKey
```

---

## External Dependencies

| Library | Used for |
|---|---|
| `yfinance` | Downloading historical price data (Adj Close) |
| `pandas` | All DataFrame operations |
| `numpy` | Numerical computations, array ops |
| `ta` (ta-lib Python port) | `RSIIndicator` in `ta.momentum` |
| `tqdm` | Progress bars |
| `quantstats` | Smart Sharpe, Sharpe, Sortino, VaR, CVaR, Volatility, Skewness, Kurtosis calculations |
| `matplotlib` | Imported but not used visibly in reviewed code (present in imports) |
| `seaborn` | Imported but not used visibly in reviewed code (present in imports) |
| `scipy.stats` | `percentileofscore` for Monte Carlo comparison |
| `scipy.optimize.minimize` | SLSQP optimizer for ERC and Smart-Sharpe weight optimization |
| `composer_tools` | Optional: converts signal names to Composer DSL code (graceful fallback if absent) |

---

## File I/O

### Reads:
- yfinance (HTTP) — max-history price data per ticker
- `cache/prices_<hash>.pkl` — pickled price DataFrame (resume cache)
- `cache/signals_<hash>.pkl` — pickled signals dict (resume cache)
- `<run_dir>/master_checkpoint.pkl` — full run state for crash recovery
- `<method>/iters/checkpoint.pkl` — per-iteration progress within each method
- `combos/combo_lowcorr_shortlist_frozen.csv` — for portfolio series building
- `combos/portfolio_weights_ssopt.csv` — optional Smart-Sharpe weights
- `analysis_inbox/*.csv` — input CSVs for `analysis_workshop.py`

### Writes (per run):
- `cache/prices_<hash>.pkl`, `cache/signals_<hash>.pkl`
- `master_checkpoint.pkl`, `checkpoint.pkl` (per method, deleted on completion)
- `README.md`, `manifest.csv`
- `aggregates/evaluation_summary.csv`
- `aggregates/method_averages.csv`
- `aggregates/quant_desk_summary.csv`
- `aggregates/<prefix>_<method>_combo_distribution_combos_only.csv` (many variants)
- `holdout/results.csv`, `holdout/combos_and_solos.csv`, `holdout/combos_only.csv`
- `walk_forward/results.csv`, `walk_forward/combos_and_solos.csv`
- `walk_forward/averages.csv`
- `walk_forward/iters/iteration_combo_logs/combos_iter<N>.csv`
- (Same structure for `expanding/` and `rolling/`)
- `combos/combo_oos_history_dynamic.csv`
- `combos/combo_quant_summary_dynamic.csv`
- `combos/combo_lowcorr_shortlist_dynamic.csv`
- `combos/combo_oos_history_frozen.csv`
- `combos/combo_quant_summary_frozen.csv`
- `combos/combo_lowcorr_shortlist_frozen.csv`
- `combos/portfolio_weights_erc_frozen.csv`
- `combos/portfolio_weights_smartsharpe_frozen.csv`
- `combos/portfolio_series_equal_weight.csv`
- `combos/portfolio_series_ssopt.csv`
- `analysis_output/filtered_results_<YYYYMMDD_HHMMSS>.csv` (analysis_workshop only)
- Optionally: `composer_signals.txt`

All formats are CSV except PKL (pickle) files.

---

## API Calls

All external HTTP calls go through `yfinance`:

| Function | Endpoint / Method | What it fetches |
|---|---|---|
| `_download_single_ticker_safe` | `yf.download(ticker, period="max", auto_adjust=False)` with fallback to `yf.Ticker(ticker).history(period="max")` | Full price history (OHLCV + Adj Close) |
| `download_prices_max_debug` | Calls `_download_single_ticker_safe` in parallel via ThreadPoolExecutor | Max history for all tickers |
| `_download_prices` | `yf.download` with start/end or period params | Price history for specified date range |
| `_yf_download_with_retry` | `yf.download` with retry | Standard date-range download |

There are no Composer API, Tiingo, or other external HTTP calls. The `composer_tools` library is a local conversion utility, not an HTTP client.

---

## Known Issues / Dead Code

1. **`_ATR` is NOT true ATR** (line ~111): The function computes rolling std of price, not Average True Range (which requires high, low, close data). The name is misleading but the approximation is documented by a comment.

2. **`_BBANDS` returns only the upper band** (line ~104): The function name implies full Bollinger Bands (upper/lower/middle), but it only returns the upper band. No lower band is available.

3. **`_is_combo_row` is defined twice** — once at line ~479 in the superset section and again at line ~1513 in the distribution section, with slightly different logic. The two versions use different detection approaches; the second one is used in `_build_combo_distribution` and `_write_combo_distribution_files`.

4. **Dead/unreachable code block** (lines ~3084–3172): A code block that calls `summarize_combo_distributions`, `build_combo_corr`, `portfolio_from_shortlist`, and `per_fold_ssopt` is syntactically indented inside a function body but is actually at module level (the `def` before it ends at ~3082). This code is never executed and references functions that are not defined anywhere in the file (likely from a deleted external module). This is dead code.

5. **`matplotlib` and `seaborn` imported but unused**: Both are imported at the top of `main.py` and `main2.py` but no matplotlib/seaborn plotting calls are present in the reviewed code.

6. **`_write_combo_distribution_files` has deeply nested loops** (lines ~1724–2044): The function has up to 8 levels of nesting and generates many nearly-identical CSV files (basic_stats, percentiles, hit_rates, stability, coverage) that largely duplicate what `_dist_summary` and `_build_combo_distribution` already produce. This appears to be an artifact of incremental feature additions.

7. **`DIAGNOSTIC H3` print statements left in production code** (lines ~2284, ~2300, etc.): Debug-level print statements prefixed with "DIAGNOSTIC H3:" are still present in `run_walk_forward_evaluation`, `run_rolling_window_evaluation`, and `run_expanding_window_evaluation`. These are not `#` comments — they execute on every run.

8. **`main2.py` vs `main.py`**: These two files are nearly identical (~5524 vs ~5070 lines). The primary confirmed difference is `max_workers=6` vs `max_workers=5` in `enrich_with_synergistic_combos`. `main2.py` appears to be an older or newer development snapshot that was never reconciled with `main.py`. Running both would produce the same results with slightly different parallelism settings.

9. **`BASE_OUTPUT_DIR`**: In `main.py` this is set to `Path(__file__).parent / "datasets"`. In `main2.py` this variable is set but no path value is shown in the first 200 lines reviewed — it likely uses a different path. (The variable exists in main2.py line 14 but the value assignment was cut off.)

10. **`_as_results_dict` and `_get_results_frame`** (lines ~52–83): These two normalization helpers are defined at module level but are not called anywhere in the reviewed code. They may be leftover from an earlier refactor.

---

## CSV Schema (Sample)

No `enhanced_backtest_results_*` dataset folders were found in the target directory at the time of inventory — the `datasets/` output folder and `cache/` folder also did not exist (no runs have been completed in this environment). The output CSV schemas are documented above from the source code analysis.

The key output files and their schemas are:

**`aggregates/evaluation_summary.csv`** columns:
```
Method, Rank, Signal, Ticker, OOS_Return, OOS_Sharpe, OOS_Max_DD,
Robustness, Train_Period, Test_Period, Total_Signals, Notes
```

**`combos/combo_quant_summary_frozen.csv`** columns:
```
Signal, Ticker, N_Iterations, Sharpe_p50, Sharpe_p10, Sharpe_p90,
Sharpe_IQR, Return_p50, Return_p10, Return_p90, MaxDD_p90,
HitRate_Positive_Sharpe
```

**`combos/portfolio_series_equal_weight.csv`** columns:
```
Date, Return, CumReturn
```

**`analysis_workshop.py` input schema** (from `combo_quant_summary_frozen.csv` or `method_averages.csv`):
```
Signal, Ticker, N_Iterations, Sharpe_p50, Sharpe_p10, Sharpe_p90,
HitRate_Positive_Sharpe, Robustness_Score_mean, Sharpe_CoV,
Median_Calmar (calculated at load time from Return_p50 / abs(MaxDD_p90))
```
