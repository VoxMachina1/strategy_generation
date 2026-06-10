# rsi_tester — Functionality Inventory

This document covers:
- `rsi_tester/run_analysis.py` (entry point — differs significantly from rsi_search)
- `rsi_tester/strategy_engine/main.py` (alternate legacy entry point)
- `rsi_tester/strategy_engine/src/` — all shared engine modules
- `rsi_tester/pathfinder/strategy_paths.py`
- `rsi_tester/strategy_filter/filter_results.py`
- `rsi_tester/strategy_inserter/strategy_inserter.py`
- `rsi_tester/setup_project.py`

The `strategy_engine/src/` modules (config_loader, data_loader, data_alignment, indicators, preconditions, signals, strategy_engine, metrics, range_tester) are **byte-for-byte identical** to the corresponding files in `rsi_search/strategy_engine/src/`. They are fully documented here; the rsi_search inventory may note "identical to rsi_tester."

---

## Overview

A modular backtesting pipeline that discovers "frontrunner" RSI signals for Composer.Trade algorithmic strategies. The pipeline:

1. **Parses** a Composer/VOXPORT strategy JSON into every possible boolean condition path leading to every leaf asset endpoint.
2. **Backtests** every combination of (signal asset × target asset × RSI threshold) against each path's precondition, measuring whether holding the target beats the path's endpoint asset when the signal fires.
3. **Filters** results to high-quality signals (win rate, trade count, benchmark performance criteria).
4. **Inserts** validated frontrunner conditions back into the strategy JSON at the exact leaf nodes, producing a modified strategy importable into Composer.

---

## Two Operational Modes

### Mode 1 — Full Pipeline (Auto Dual-Pass)

Selected by entering `1` (or pressing Enter) at startup. Runs two complete backtest passes automatically:
- **Pass 1 (Overbought):** `signal_operator = ">"`, thresholds 50.0 → 99.0
- **Pass 2 (Oversold):** `signal_operator = "<"`, thresholds 1.0 → 50.0

After both passes complete, automatically invokes `filter_results.py` to produce `filtered.csv`. The `signal_operator` and `threshold_start/end` fields in `template.yaml` are **ignored** in this mode (overridden by hardcoded ranges).

### Mode 2 — Single Operator Pass

Selected by entering `2`. Runs exactly one pass using all settings from `template.yaml` as-is. Produces a raw timestamped CSV. No filtering is applied. Useful for manual exploration of specific threshold ranges or operators.

---

## Entry Points

### `run_analysis.py` — primary entry point, line 1
**Command:** `python run_analysis.py`

**Interactive prompts:**
1. `Enter choice (1/2, default 1):` — selects Mode 1 (full pipeline) or Mode 2 (single pass)

**What it does (Mode 1):**
- Parses strategy JSON → extracts all paths via `extract_paths()`
- Loads API keys from `.env`
- Checks/refreshes price data for all tickers via Tiingo
- Runs Pass 1 (overbought) and Pass 2 (oversold) — each enumerates every path × target × signal combination
- For each combo, runs `run_pipeline()` which iterates over all thresholds
- Writes each pass to a timestamped CSV in `strategy_engine/results/`
- Automatically calls `filter_results.run_filter()` with the two pass paths
- Outputs next-step instructions

**What it does (Mode 2):**
- Same setup, but runs one pass using template settings, writes one CSV, no filtering

**Output files (Mode 1):**
- `strategy_engine/results/{strategy_name}_{timestamp}.csv` (×2, one per operator)
- `strategy_filter/filtered.csv`
- `strategy_filter/filter_summary.txt`

---

### `strategy_engine/main.py` — legacy alternate entry point, line 77
**Command:** `python strategy_engine/main.py --config config/strategy_config.yaml`

**What it does:** Manual config-file-driven discovery loop. Reads a YAML config (not template.yaml), uses `generate_asset_combinations()` from range_tester, iterates over all combos, builds the pipeline per combo, saves one aggregated CSV. Does not parse a strategy JSON or use paths — takes signal_assets, target_assets, benchmark_asset, filter_assets, and preconditions all from the config file.

This is the older/simpler engine. `run_analysis.py` is the current production entry point.

**Args:** `--config` / `-c` (path to YAML, default `config/strategy_config.yaml`)

---

### `pathfinder/strategy_paths.py` — standalone path extractor
**Command:** `python pathfinder/strategy_paths.py`

Reads `pathfinder/strategy.json`, walks the entire strategy tree, emits all paths to stdout and writes `pathfinder/paths.txt`.

---

### `strategy_filter/filter_results.py` — standalone filter
**Command:** `python strategy_filter/filter_results.py`

Auto-discovers the two most recent CSVs in `strategy_engine/results/`, validates operator split, applies filters, writes `strategy_filter/filtered.csv`.

---

### `strategy_inserter/strategy_inserter.py` — standalone inserter
**Command:**
```
python strategy_inserter/strategy_inserter.py [filtered_csv] [--input json] [--output json] [--period N] [--dry-run]
```

Reads `strategy_filter/filtered.csv` (default) and `pathfinder/strategy.json`, inserts frontrunner logic, writes `strategy_inserter/strategy_modified.json` and `strategy_inserter/insertion_log.json`.

---

### `setup_project.py` — scaffolding utility
**Command:** `python setup_project.py`

Creates the initial `strategy_engine/` directory tree with empty placeholder files. One-time setup tool, no ongoing use.

---

## Functions & Classes (exhaustive)

### run_analysis.py

**Module-level constants:**
- `MODE1_GT_START = 50.0`, `MODE1_GT_END = 99.0` — overbought threshold range
- `MODE1_LT_START = 1.0`, `MODE1_LT_END = 50.0` — oversold threshold range
- `EXTRA_SIGNAL_ASSETS` — hardcoded list of 15 tickers always tested as signal assets regardless of what appears in the strategy (SPY, SPYV, IOO, VTV, QQQ, QQQE, XLF, XLK, XLE, XLY, XLP, TLT, USO, CORP, GLD)
- `META_COLS`, `METRIC_COLS`, `ALL_COLS` — CSV column order definitions

**`extract_tickers_from_precondition(engine_precondition)`** — line 128
- **Parameters:** `engine_precondition` (str or None)
- **What it does:** Regex-scans the engine precondition string for ticker patterns (`[A-Z][A-Z0-9]*_[A-Za-z_0-9]+`). Extracts the leading ticker part.
- **Returns:** `set` of ticker strings (e.g., `{'SPY', 'TLT'}`)

**`build_config(path, signal_asset, target_asset, filter_assets, benchmark_asset, operator, threshold_start, threshold_end)`** — line 142
- **Parameters:** `path` (dict from `extract_paths`), asset strings, operator str, threshold floats
- **What it does:** Constructs a config dict for `load_config(config_dict=...)` compatible with the engine, merging template settings with per-path/per-asset values.
- **Returns:** dict (config schema — see Config YAML Schema section below)

**`run_pipeline(config, path_meta)`** — line 170
- **Parameters:** `config` (dict), `path_meta` (path dict from `extract_paths`)
- **What it does:** Runs the full engine pipeline for one (signal, target, operator) combination across all thresholds. Builds master dataframe, adds signal indicator, parses and adds precondition indicators, evaluates preconditions, calculates returns, then loops over all thresholds calling generate_signals → filter_date_range → calculate_strategy_returns → calculate_equity_curves → calculate_metrics.
- **Returns:** list of metrics dicts (one per threshold)

**`run_pass(path_results, api_keys, all_target_assets, operator, threshold_start, threshold_end, label)`** — line 261
- **Parameters:** `path_results` (list of path dicts), `api_keys` (list), `all_target_assets` (list), operator/threshold settings, `label` (str for logging)
- **What it does:** The outer loop for one operator pass. For each path, determines signal assets (EXTRA_SIGNAL_ASSETS union tickers in precondition, minus endpoint), filter assets (precondition tickers plus endpoint), and target candidates (all_target_assets minus endpoint). Calls `run_pipeline()` for each (target × signal) combo. Writes all results to a timestamped CSV.
- **Returns:** `(all_rows, total_runs, failed_runs, output_path)`
- **Side effects:** Writes CSV to `strategy_engine/results/`

**`main()`** — line 341
- Orchestrates mode selection, strategy parsing, data freshness, pass execution, and filter pipeline.

---

### strategy_engine/main.py (legacy)

**`save_results(results_list, results_dir, run_id, mode)`** — line 20
- **Parameters:** `results_list` (list of dicts), `results_dir` (str or Path), `run_id` (str), `mode` (str: `"per_strategy"` or `"master"`)
- **What it does:** Converts results to DataFrame, adds `run_id` column. In `"master"` mode: appends to `results/results.csv`. In `"per_strategy"` mode: writes `results/{run_id}.csv`.
- **Returns:** `True`
- **Side effects:** Writes CSV file

**`parse_and_add_precondition_indicators(df, precondition_string)`** — line 46
- **Parameters:** `df` (pd.DataFrame), `precondition_string` (str)
- **What it does:** Scans the precondition string for patterns like `SPY_RSI_10` or `signal_SMA_200`. For each match, calls `add_indicator()` if the column doesn't already exist. Warns if the required `{role}_close` column is missing.
- **Returns:** df (modified)

**`main()`** — line 77
- Parses `--config` arg, loads config, checks freshness, generates combinations, runs strategy loop, saves results.

---

### strategy_engine/src/config_loader.py

**`load_config(config_filename, config_dict)`** — line 7
- **Parameters:** `config_filename` (str, default `"config/strategy_config.yaml"`), `config_dict` (dict or None)
- **What it does:** Locates `strategy_engine/` root dir from `__file__`. Loads `.env` from that root. Reads `TIINGO_API_KEYS` env var; parses as JSON array or comma-separated list. If `config_dict` is provided, skips file loading and returns it directly. Otherwise loads YAML from disk.
- **Returns:** `(config_dict, api_keys_list)`
- **Side effects:** Calls `dotenv.load_dotenv()`

---

### strategy_engine/src/data_loader.py

**`get_latest_tiingo_date(api_keys)`** — line 12
- **Parameters:** `api_keys` (list of str)
- **What it does:** GETs SPY prices for the last 10 days from Tiingo, returns the date of the last entry as a `YYYY-MM-DD` string. Rotates through keys on failure.
- **Returns:** str (date)
- **Side effects:** HTTP GET to `https://api.tiingo.com/tiingo/daily/SPY/prices`

**`download_ticker_data(ticker, api_keys, data_dir)`** — line 36
- **Parameters:** `ticker` (str), `api_keys` (list), `data_dir` (Path)
- **What it does:** GETs full history from 1900-01-01 via Tiingo. Rotates keys on failure. Converts to DataFrame with `date` (str) and `close` (adjClose). Saves to `{data_dir}/{ticker}.csv`.
- **Returns:** `True`
- **Side effects:** HTTP GET to `https://api.tiingo.com/tiingo/daily/{ticker}/prices`; writes CSV

**`check_freshness_and_update(tickers, api_keys, data_dir)`** — line 73
- **Parameters:** `tickers` (list), `api_keys` (list), `data_dir` (Path)
- **What it does:** For each ticker, reads its CSV (if it exists), compares the max date against the latest Tiingo market date. If outdated or missing, calls `download_ticker_data()` to rebuild the full history.
- **Returns:** None
- **Side effects:** May download and overwrite CSVs; prints status per ticker

---

### strategy_engine/src/data_alignment.py

**`load_ticker_csv(ticker, data_dir)`** — line 5
- **Parameters:** `ticker` (str), `data_dir` (Path)
- **What it does:** Reads `{data_dir}/{ticker}.csv`, parses `date` as datetime, sorts ascending.
- **Returns:** pd.DataFrame with columns `['date', 'close']`

**`build_master_dataframe(signal_ticker, target_ticker, benchmark_ticker, data_dir, filter_assets)`** — line 19
- **Parameters:** ticker strings for each role; `filter_assets` (list, default `[]`)
- **What it does:** Loads CSVs for signal, target, benchmark (with role-prefixed column names: `signal_close`, `target_close`, `benchmark_close`). For each filter asset, loads with ticker-named column (e.g., `SPY_close`). Inner-joins all on `date`. Drops any rows with NaN.
- **Returns:** pd.DataFrame with columns `['date', 'signal_close', 'target_close', 'benchmark_close', '{TICKER}_close', ...]`

---

### strategy_engine/src/indicators.py

**`calculate_sma(series, period)`** — line 11
- **Parameters:** pd.Series, int
- **Returns:** pd.Series — simple moving average (rolling mean)

**`calculate_ema(series, period)`** — line 15
- **Parameters:** pd.Series, int
- **Returns:** pd.Series — EMA using `ewm(span=period, adjust=False)`

**`calculate_rsi(series, period)`** — line 19
- **Parameters:** pd.Series (price), int (period)
- **What it does:** Computes Wilder's RSI using EMA with alpha=1/period. First `period` rows set to NaN. Handles zero-average-loss edge case (RSI=100).
- **Returns:** pd.Series of RSI values (0–100)

**`calculate_cumret(series, period)`** — line 45
- **Parameters:** pd.Series (price), int (period)
- **Returns:** pd.Series — rolling percentage return over N periods (`pct_change(periods=N) * 100`)

**`add_indicator(df, asset_role, indicator_name, period)`** — line 49
- **Parameters:** df (pd.DataFrame), `asset_role` (str, e.g. `"signal"` or `"SPY"`), `indicator_name` (str: RSI/SMA/EMA/CUMRET), `period` (int)
- **What it does:** Dispatches to the appropriate calculate_* function. Reads `{asset_role}_close` column. Writes `{asset_role}_{INDICATOR}_{period}` column.
- **Returns:** modified df (copy)
- **Raises:** `ValueError` if price column missing or indicator name unknown

---

### strategy_engine/src/preconditions.py

**`evaluate_preconditions(df, precondition_string)`** — line 12
- **Parameters:** df (pd.DataFrame), `precondition_string` (str or None)
- **What it does:** If string is empty/None/`"none"`/`"[]"`, sets `precondition_pass = 1` for all rows. Otherwise evaluates the string using `df.eval(precondition_string)` — this allows any valid pandas expression referencing DataFrame column names (e.g., `"SPY_RSI_10 > 80 and QQQ_RSI_10 > 80"`). Sets `precondition_pass` to 1 where the expression is True, 0 elsewhere.
- **Returns:** modified df (copy) with `precondition_pass` column added

**Precondition expression format:**
Any valid `pandas.DataFrame.eval()` expression referencing columns in the master DataFrame. Columns are named `{TICKER}_{INDICATOR}_{PERIOD}` (e.g., `SPY_RSI_10`) or `{role}_close`. Operators: standard Python boolean operators (`and`, `or`, `not`, `>`, `<`, `>=`, `<=`, `==`, `!=`). Multiple conditions joined with `and`.

---

### strategy_engine/src/signals.py

**`generate_signals(df, indicator_col, operator_str, threshold)`** — line 17
- **Parameters:** df, `indicator_col` (str, e.g. `"signal_RSI_10"`), `operator_str` (str: `>`, `<`, `>=`, `<=`, `==`, `!=`), `threshold` (float)
- **What it does:** Applies the operator to `df[indicator_col]` vs `threshold`. ANDs the result with `precondition_pass` (if present in df, defaults to all-True if absent). Writes result to `signal_active` column (1=active/target, 0=inactive/benchmark).
- **Returns:** modified df (copy)

---

### strategy_engine/src/strategy_engine.py

**`calculate_asset_returns(df)`** — line 19
- **Parameters:** df
- **What it does:** Computes forward 1-day return for target and benchmark: `pct_change().shift(-1)`. Drops rows where either is NaN (i.e., drops the last row).
- **Returns:** modified df with `target_return` and `benchmark_return` columns

**`calculate_strategy_returns(df, slippage_bps)`** — line 29
- **Parameters:** df, `slippage_bps` (float, default 0.0)
- **What it does:** Sets `strategy_return` to `target_return` where `signal_active == 1`, else `benchmark_return`. If slippage > 0, subtracts `slippage_bps/10000` × trade_multiplier from each day. Trade multiplier is 1 on first entry, 2 on state changes (sell one, buy another), 0 when no change.
- **Returns:** modified df with `strategy_return` column

**`filter_date_range(df, start_date, end_date)`** — line 50
- **Parameters:** df, `start_date` (str or None), `end_date` (str or None)
- **What it does:** Filters df rows to `start_date <= date <= end_date`. Either bound can be None to be unbounded.
- **Returns:** filtered df (reset index)

**`calculate_equity_curves(df, initial_capital)`** — line 62
- **Parameters:** df, `initial_capital` (float, default 1.0)
- **What it does:** Computes `strategy_equity` and `benchmark_equity` as compounding cumulative product of `(1 + return)`.
- **Returns:** modified df with two equity curve columns

---

### strategy_engine/src/metrics.py

**`calculate_metrics(df, strategy_params)`** — line 21
- **Parameters:** df (with `signal_active`, `strategy_return`, `benchmark_return`, `strategy_equity`, `benchmark_return` columns), `strategy_params` (dict — passed through and augmented)
- **What it does:** Computes all performance metrics for the backtest period:
  - `Total_Trades` — count of days `signal_active == 1`
  - `Win_Rate` — fraction of active days where `strategy_return > 0`
  - `Avg_Return` — mean of `strategy_return` on active days
  - `Median_Return` — median of `strategy_return` on active days
  - `Benchmark_Avg_Return` / `Benchmark_Median_Return` — benchmark stats on active days only
  - `Total_Return` — `final_equity - 1.0`
  - `Annualized_Return` — `final_equity^(252/total_days) - 1`
  - `Sharpe_Ratio` — `(mean_excess_return / std_return) * sqrt(252)` using `risk_free_rate` from params
  - `Sortino_Ratio` — uses only downside returns in denominator
  - `Calmar_Ratio` — `annualized_return / abs(max_drawdown)`
  - `Max_Drawdown` — minimum of `(equity / rolling_max_equity) - 1`
  - `Final_Equity` — last value of `strategy_equity`
  - `Avg_Hold_Days` — `total_active_days / number_of_signal_on_streaks`
- **Returns:** dict merging `strategy_params` with all computed metrics

---

### strategy_engine/src/range_tester.py

**`generate_threshold_range(start, end, step)`** — line 23
- **Parameters:** floats
- **Returns:** list of floats from `start` to `end` inclusive, spaced by `step` (uses `np.arange` with half-step epsilon to include endpoint, rounded to 4 decimals)

**`generate_asset_combinations(signal_assets, target_assets, benchmark_asset)`** — line 32
- **Parameters:** lists of ticker strings, benchmark string
- **What it does:** Cross-product of signal × target. Enforces `signal_asset != target_asset` constraint.
- **Returns:** list of dicts `{"signal_asset": str, "target_asset": str, "benchmark_asset": str}`

**`run_threshold_range_tests(df, base_params, thresholds, sig_col, sig_op, start_date, end_date)`** — line 48
- **Parameters:** pre-built df (with indicator and precondition columns), base_params dict, threshold list, signal column name, operator string, date strings
- **What it does:** For each threshold: calls `generate_signals()`, `filter_date_range()`, `calculate_strategy_returns()` (with slippage from base_params), `calculate_equity_curves()`, `calculate_metrics()`. Collects and returns all results.
- **Returns:** list of metrics dicts

---

### pathfinder/strategy_paths.py

**Module-level constants:**
- `STRATEGY_JSON` — `pathfinder/strategy.json`
- `PATHS_TXT` — `pathfinder/paths.txt`
- `FN_LABELS` — map from Composer internal fn names to human labels (RSI, CumRet, MA, Price)
- `COMPARATOR_LABELS` — map from Composer comparator strings (gt, lt, gte, lte, eq, neq) to symbols
- `COMPARATOR_NEGATIONS` — map to negated symbols (for else-branch conditions)
- `FN_ENGINE` — map from Composer fn names to engine column label parts (RSI, CumRet, SMA, None for price)

**`_get_window(node, prefix)`** — line 74
- Extracts window parameter from a Composer JSON node (checks `{prefix}-fn-params.window` then `{prefix}-window-days`)
- **Returns:** str or None

**`_format_side(node, prefix)`** — line 88 — human-readable side of a condition
**`_format_condition(if_child)`** — line 98 — full human-readable condition string
**`_format_negated_condition(if_child)`** — line 105 — negated form for else branches
**`_engine_side(node, prefix)`** — line 116 — engine DataFrame column name for one side (e.g., `TLT_RSI_20` or `SPY_close`)
**`_engine_condition(if_child)`** — line 139 — full engine-compatible condition string
**`_engine_negated_condition(if_child)`** — line 146 — negated engine condition
**`_filter_sort_fn(node)`** — line 157 — extracts (fn_raw, window_str) from a filter node
**`_is_portfolio_filter(node)`** — line 165 — returns True if filter node's children are all "group" steps (portfolio filter, not asset-selection filter)

**`_get_ticker_and_id(child)`** — line 170
- **Parameters:** a child node dict
- **What it does:** Handles two cases: bare `"asset"` step (returns ticker + node id directly) or `"wt-cash-equal"` wrapper inserted by strategy_inserter (walks into else branch to find original asset ticker, returns ticker + wrapper's id).
- **Returns:** `(ticker, node_id)` or `(None, None)`

**`_expand_filter(node, conditions, engine_conds, sub_strategy, results)`** — line 199
- **Parameters:** filter node, accumulated conditions lists, sub-strategy name, results accumulator
- **What it does:** Expands a top-1 asset-selection filter (exactly 2 assets required) into 2 pairwise paths. Each path gets a condition `winner_FN(window) > loser_FN(window)`. Only `select-n=1` is supported; 2-asset filters only.
- **Returns:** None (appends to results in place)
- **Raises:** `NotImplementedError` for select-n > 1 or non-2-asset filters

**`walk(node, conditions, engine_conds, sub_strategy, results)`** — line 255
- **Parameters:** current JSON node, accumulated human-readable conditions list, accumulated engine conditions list, current sub-strategy name, results list
- **What it does:** Recursive tree walker. Dispatches on `node["step"]`:
  - `root`, `wt-cash-equal`, `wt-cash-specified` — recurse into children
  - `group` — sets sub_strategy name from first group, recurses
  - `asset` — appends a result dict (leaf reached)
  - `filter` — routes to `_is_portfolio_filter` or `_expand_filter`
  - `if` — splits into positive branches (condition added) and else branches (negated condition added); if no else, records `UNALLOCATED`
  - `if-child` — recurse into children
  - unknown step — prints warning to stderr, recurses
- **Returns:** None (accumulates into results)

**`format_path(path)`** — line 337 — formats one result dict as a readable one-line string

**`extract_paths(strategy_json)`** — line 352 — **public API**
- **Parameters:** loaded strategy JSON dict
- **What it does:** Calls `walk()` starting from the root node.
- **Returns:** list of path dicts, each containing:
  ```python
  {
    "sub_strategy":        str,   # name of the top-level group/sleeve
    "conditions":          list[str],  # human-readable condition chain
    "engine_conds":        list[str],  # engine-compatible condition chain
    "engine_precondition": str,   # engine_conds joined with ' and '
    "endpoint":            str,   # leaf asset ticker
    "node_id":             str,   # UUID of the leaf asset node in the JSON
  }
  ```

**`main()`** — line 367 — standalone entry point: reads strategy.json, extracts paths, prints sorted by endpoint, writes paths.txt

---

### strategy_filter/filter_results.py

**Hardcoded filter criteria:**
- `Win_Rate > 0.75`
- `Total_Trades > 20`
- `Benchmark_Median_Return < 0`

**`find_two_most_recent_csvs(results_dir)`** — line 51
- Scans for `*.csv` in results_dir excluding `filtered*.csv` and `.~lock` files, sorted by mtime descending.
- **Returns:** `(most_recent_path, second_most_recent_path)`
- **Side effects:** `sys.exit(1)` if fewer than 2 CSVs found

**`find_two_specific_csvs(path_gt, path_lt)`** — line 72
- Accepts two explicit Path objects (bypasses discovery and validation).
- **Returns:** `(df_gt, df_lt, path_gt, path_lt)`

**`validate_and_assign(csv_a, csv_b)`** — line 88
- Loads both CSVs, confirms each has `signal_operator` column, confirms each has exactly one operator value, confirms one is `">"` and the other is `"<"`.
- **Returns:** `(df_gt, df_lt, path_gt, path_lt)` ordered so gt is the `>` file
- **Side effects:** `sys.exit(1)` on any validation failure

**`apply_filters(df, label)`** — line 140
- Applies the three hardcoded filter criteria.
- **Returns:** filtered DataFrame

**`run_filter(path_gt, path_lt)`** — line 165 — **public API**
- **Parameters:** `path_gt` and `path_lt` (optional Path objects — if provided, skips discovery/validation)
- **What it does:** Full filter pipeline. If paths provided, reads them directly (Mode 1 path). If not, auto-discovers and validates. Applies filters to both, concatenates, writes `strategy_filter/filtered.csv` and `strategy_filter/filter_summary.txt`.
- **Returns:** `OUTPUT_CSV` path

**`main()`** — line 254 — calls `run_filter()` with no args (standalone mode)

---

### strategy_inserter/strategy_inserter.py

**`new_id()`** — line 80 — returns a new random UUID string

**`most_inclusive_threshold(thresholds, operator)`** — line 109
- **Parameters:** list of float thresholds, operator string
- **What it does:** Returns the threshold that fires most often: minimum for `>` / `>=`, maximum for `<` / `<=`.
- **Returns:** float

**`is_tautology(thresh_gt, thresh_lt)`** — line 127
- **Parameters:** two floats
- **What it does:** Returns `True` if `thresh_gt < thresh_lt` — meaning `RSI > thresh_gt OR RSI < thresh_lt` is always true (the two ranges overlap and cover all possible RSI values).
- **Returns:** bool
- **Example:** `is_tautology(16, 18.5) == True` (always true since any RSI either > 16 or < 18.5); `is_tautology(80, 20) == False` (RSI values 20–80 are uncovered)

**`build_asset_node(ticker)`** — line 143 — constructs a Composer `"asset"` step dict with new UUID

**`_fmt_threshold(value)`** — line 153 — formats float as int string if whole number, otherwise decimal string

**`build_if_node(signal_asset, operator, threshold, ind_period, target_tickers, original_asset_node)`** — line 157
- **What it does:** Builds a Composer `"if"` step dict with a true branch (RSI condition → target asset nodes) and an else branch (deepcopy of original asset node). Uses `FN_MAP` and `COMPARATOR_MAP` to produce Composer-compatible field names.
- **Returns:** dict (Composer if node)

**`build_wt_cash_equal_wrapper(if_nodes)`** — line 205 — wraps a list of if nodes in a `"wt-cash-equal"` step dict

**`build_if_nodes_for_group(group_df, original_asset_node, ind_period)`** — line 219
- **Parameters:** DataFrame of filtered results for one (sub_strategy, conditions, endpoint) group, the original leaf asset node dict, RSI period int
- **What it does (5-step algorithm):**
  1. For each `(signal_asset, operator, target_asset)` triple, find the most inclusive threshold across all tested thresholds
  2. Tautology check: for each signal_asset that appears with both `>` and `<`, check if the most inclusive thresholds overlap. If so, skip that signal_asset entirely (logs warning).
  3. Group targets by `(signal_asset, operator, threshold)` — multiple targets sharing same key become siblings in one if-node's true branch
  4. For each unique `(signal_asset, operator, threshold)`, call `build_if_node()` with targets sorted by descending `Median_Return`
  5. Sort if-nodes: most targets first, then most inclusive threshold within same target count
- **Returns:** list of if-node dicts (sorted, tautologies excluded)

**`build_node_index(tree)`** — line 311 — recursive walker that builds `{node_id: node_dict}` index
**`build_parent_index(tree)`** — line 321 — recursive walker that builds `{child_id: (parent_node, child_index)}` index

**`normalise_conditions(cond_str)`** — line 339 — strips, uppercases, and collapses whitespace in a condition string

**`build_path_to_nodeid_map(strategy_tree)`** — line 347
- Calls `extract_paths()` on the tree, builds a lookup dict keyed by `(sub_strategy, normalised_conditions_string, endpoint_upper)` → node_id
- **Returns:** dict

**`load_and_group_candidates(csv_path)`** — line 367
- Reads filtered CSV. Validates `signal_operator` column exists and contains only `{>, <, >=, <=}`. Groups by `(sub_strategy, conditions, endpoint)`.
- **Returns:** list of spec dicts `{"sub_strategy", "conditions", "endpoint", "group_df"}`
- **Side effects:** `sys.exit(1)` on validation failures

**`insert_frontrunners(strategy_tree, csv_path, ind_period, dry_run)`** — line 413 — **main insertion pipeline**
- **Parameters:** loaded strategy JSON dict, path to filtered CSV, RSI period int, dry_run bool
- **What it does:**
  1. Deep-copies the tree
  2. Builds path map from the *original* strategy.json (always re-read from disk for stable node IDs)
  3. Builds parent and node indexes on the working copy
  4. Loads and groups candidates from CSV
  5. For each group: looks up the node_id in the path map, builds if-nodes via `build_if_nodes_for_group()`, wraps them in `wt-cash-equal`, replaces the leaf node in the parent's children list (unless dry_run)
  6. Logs all insertions and skips
- **Returns:** `(modified_tree, log_list)`

**`main()`** — line 522 — CLI entry point; parses args, loads JSON and CSV, calls `insert_frontrunners()`, writes output JSON and log

---

### setup_project.py

**`create_project_structure()`** — line 3
- Creates `strategy_engine/{data,results,config,src}/` directories and touches all src Python files and `main.py`. Prints verification output.
- **Returns:** None
- **Side effects:** Creates directories and empty files

---

## Config YAML Schema (template.yaml)

```yaml
# Candidate frontrunner target assets — tested as the asset to hold when signal fires
target_assets: ["BIL", "UVXY"]     # list of str

# Only used in strategy_engine/main.py and Mode 2 of run_analysis.py:
benchmark_asset: "SPY"              # str — manual runs only
signal_assets: ["QQQ", "XLF"]      # list of str — manual runs only
filter_assets: ["SPY"]              # list of str — tickers needed for preconditions

indicator: "RSI"                    # str — RSI is the only production use
indicator_period: 10                # int

# Mode 2 / manual runs only (ignored in Mode 1):
signal_operator: ">"                # str: ">" or "<"
threshold_start: 50.0               # float
threshold_end: 80.0                 # float

threshold_step: 0.5                 # float — used in both modes

slippage_bps: 1.0                   # float — basis points (1.0 = 0.01% per trade leg)
risk_free_rate: 0.0                 # float — annual, used in Sharpe/Sortino

date_range:
  start: "2015-01-01"               # str YYYY-MM-DD
  end: "2026-03-15"                 # str YYYY-MM-DD

# Only used in strategy_engine/main.py:
preconditions: ""                   # str — pandas eval-compatible boolean expression
results_mode: "per_strategy"        # str: "per_strategy" or "master"
```

---

## Data Structures

### Master DataFrame (output of `build_master_dataframe`)
```
date              datetime64
signal_close      float64        # price of signal asset
target_close      float64        # price of target asset
benchmark_close   float64        # price of benchmark asset
{TICKER}_close    float64        # one per filter_asset
```

### After indicators added (`add_indicator`):
```
{role}_{INDICATOR}_{period}   float64   # e.g. signal_RSI_10, SPY_SMA_200
```

### After preconditions (`evaluate_preconditions`):
```
precondition_pass   int (0 or 1)
```

### After signals (`generate_signals`):
```
signal_active   int (0 or 1)
```

### After strategy engine (`calculate_asset_returns`, `calculate_strategy_returns`, `calculate_equity_curves`):
```
target_return      float64   # forward 1-day pct return of target
benchmark_return   float64   # forward 1-day pct return of benchmark
strategy_return    float64   # return of active strategy (target or benchmark, minus slippage)
strategy_equity    float64   # compounding equity curve (starts at 1.0)
benchmark_equity   float64   # compounding equity curve (starts at 1.0)
```

### extract_paths() output dict per path:
```python
{
  "sub_strategy":        str,   # top-level group name (e.g. "Bond Compares")
  "conditions":          list[str],   # ["RSI(TLT, 20) > RSI(PSQ, 20)", ...]
  "engine_conds":        list[str],   # ["TLT_RSI_20 > PSQ_RSI_20", ...]
  "engine_precondition": str,   # engine_conds joined with ' and '
  "endpoint":            str,   # leaf asset ticker (the strategy's target/benchmark)
  "node_id":             str,   # UUID of the leaf node in the strategy JSON
}
```

### Insertion log entry (per insertion in `insert_frontrunners`):
```python
{
  "sub_strategy":  str,
  "conditions":    str,   # space-joined conditions string
  "endpoint":      str,
  "node_id":       str,
  "target_assets": list[str],
  "signal_assets": list[str],
  "operators":     list[str],
  "if_node_count": int,
}
```

---

## External Dependencies

| Library | Usage |
|---|---|
| `pandas` | All DataFrames, eval-based precondition evaluation, CSV I/O |
| `numpy` | Array math (RSI, equity curves, metrics) |
| `requests` | Tiingo HTTP API calls |
| `pyyaml` | Loading template.yaml and config files |
| `python-dotenv` | Loading `.env` for API keys |
| `pathlib` | All path handling |

---

## File I/O

| File | Format | Read/Write | Notes |
|---|---|---|---|
| `pathfinder/strategy.json` | JSON | Read | Composer/VOXPORT strategy export |
| `pathfinder/paths.txt` | Text | Write | Human-readable path list (standalone mode) |
| `strategy_engine/.env` | Key=value | Read | `TIINGO_API_KEYS=key1,key2,...` |
| `strategy_engine/config/template.yaml` | YAML | Read | Engine settings |
| `strategy_engine/data/{TICKER}.csv` | CSV | Read/Write | `date,close` — auto-managed price cache |
| `strategy_engine/results/{name}_{timestamp}.csv` | CSV | Write | Raw backtest results per pass |
| `strategy_filter/filtered.csv` | CSV | Write | Combined filtered results |
| `strategy_filter/filter_summary.txt` | Text | Write | Human-readable filter summary |
| `strategy_inserter/strategy_modified.json` | JSON | Write | Modified strategy for Composer import |
| `strategy_inserter/insertion_log.json` | JSON | Write | Log of all insertions made |

---

## API Calls

| Endpoint | Method | What it fetches |
|---|---|---|
| `https://api.tiingo.com/tiingo/daily/{ticker}/prices` | GET | Full historical daily adjusted close prices; also used with `startDate=last-10-days` to detect the latest trading date |

---

## Mode 1 Filter Criteria (hardcoded in filter_results.py)

Applied to both the overbought and oversold result sets independently before combining:
- `Win_Rate > 0.75` — signal must beat the endpoint asset more than 75% of active days
- `Total_Trades > 20` — at least 21 signal-active days in the backtest window
- `Benchmark_Median_Return < 0` — the endpoint asset (benchmark) must have a negative median return on the days the signal fires (i.e., the signal fires precisely when the benchmark is hurting)

---

## Known Issues / Dead Code

- `strategy_engine/main.py` — the older config-file-driven entry point. Still functional but superseded by `run_analysis.py` for production use. Not deleted; may be useful for manual exploration without a strategy JSON.
- `setup_project.py` — one-time scaffolding script, no ongoing use. Kept in root.
- **`find_two_specific_csvs()`** in filter_results.py — defined but only called internally by `run_filter()` in Mode 1 path; the standalone entry point (`main()`) uses the discovery path, making `find_two_specific_csvs` a named but not directly-callable helper.
- **Filter criteria are hardcoded** — a TODO comment in filter_results.py (`# TODO: make configurable`) notes that WIN_RATE_MIN, TOTAL_TRADES_MIN, BENCHMARK_MEDIAN_RETURN_MAX are constants, not config-driven.
- **`_is_portfolio_filter()` limitation** — strategy_paths.py silently passes portfolio-filter children through to `walk()` individually (line 282–283), which may miss filter-level conditions in complex portfolio-filter setups.
- **`_expand_filter()` only handles 2-asset top-1 filters** — raises `NotImplementedError` for any other configuration (select-n > 1 or != 2 resolvable children).
