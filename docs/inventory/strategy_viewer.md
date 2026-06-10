# Strategy Viewer — Fuzz Tester: Complete Functionality Inventory

Generated: 2026-06-08  
Source: `C:\Python Projects\strategy_viewer\fuzz_tester\`

---

## Overview

The fuzz tester takes a Composer.Trade strategy exported as JSON, extracts every IF-condition found in the strategy tree, runs a parameter sweep around each condition's fitted values, and produces a single self-contained HTML report. The report answers two questions for each condition:

1. **Is this condition fragile?** (parameter-sensitive — does a small change in period/threshold destroy performance?)
2. **Is this condition tail-dependent?** (driven by a handful of extreme return days, not consistent daily outperformance?)

The sweep engine measures win rate of the allocated asset vs BIL (cash proxy) on signal-fired days. The HTML report contains interactive heatmaps, a fragility-ranked sidebar, equity curve charts, signal overlay charts, indicator value charts, and a per-cell stat panel.

---

## Entry Points

### Interactive mode
```
python fuzz_tester.py
```
Prompts for:
- Strategy JSON path (default: `pathfinder/strategy.json`)
- RSI fuzz range % (default: 30)
- MA fuzz range % (default: 30)
- CumRet fuzz range % (default: 30)
- Price/MA cross fuzz range % (default: 30)
- MaxDD fuzz range % (default: 30)
- Threshold step size (default: 0.5)
- Period step size integer (default: 1)
- Primary strategy asset for vs-primary stat (default: TQQQ)
- Start date (default: 2015-01-01)
- End date (default: 2026-03-15)

### Non-interactive mode
```
python fuzz_tester.py pathfinder/bestsignals3.json
```
All parameters default; uses hardcoded defaults identical to interactive defaults.

### Output
A file named `fuzz_report_{json_stem}_{YYYYMMDD_HHMMSS}.html` in the fuzz_tester directory.

---

## Functions & Classes (Exhaustive)

### `fuzz_tester.py`

#### `calculate_maxdd(series, period)` — line ~41
- **Parameters:** `series` (pd.Series of prices), `period` (int, rolling window)
- **Returns:** pd.Series of rolling max drawdown values (positive %, 0–100+)
- **What it does:** Applies a rolling window of `period` days. For each window, computes the order-aware running-peak-to-trough drawdown: `(peak - price) / peak * 100`. Takes the max drawdown over the window. Uses `np.maximum.accumulate` on raw array values.
- **Side effects:** None

#### `calculate_mareturn(series, period)` — line ~57
- **Parameters:** `series` (pd.Series of prices), `period` (int)
- **Returns:** pd.Series of rolling mean daily returns (decimal, e.g. 0.001 not 0.1%)
- **What it does:** `series.pct_change().rolling(window=period).mean()`
- **Side effects:** None

#### `_get_window(node, prefix)` — line ~86
- **Parameters:** `node` (dict, JSON node), `prefix` (str, "lhs" or "rhs")
- **Returns:** int or None — the window/period for the given side
- **What it does:** Checks `{prefix}-fn-params.window`, then falls back to `{prefix}-window-days`.

#### `_parse_side(node, prefix)` — line ~93
- **Parameters:** `node` (dict, if-child JSON node), `prefix` ("lhs" or "rhs")
- **Returns:** dict — `{"type": "fixed", "value": float}` or `{"type": "indicator", "fn_raw": str, "fn_label": str, "ticker": str, "window": int|None}`
- **What it does:** Parses one side of a direct if-child condition. If `prefix == "rhs"` and `rhs-fixed-value?` is truthy, returns a fixed dict. Otherwise reads `{prefix}-fn`, `{prefix}-val`, and window params.

#### `_parse_side_from_condition_spec(side_spec, ticker_override=None)` — line ~105
- **Parameters:** `side_spec` (dict, Composer condition payload side), `ticker_override` (str|None)
- **Returns:** same side dict format as `_parse_side`
- **What it does:** Parses a side from the nested condition-payload format (used for `binary-compound`/`binary` shapes). Handles `constant` key for fixed values; handles `%` ticker placeholder by substituting `ticker_override`.

#### `_extract_atomic_conditions(if_child)` — line ~122
- **Parameters:** `if_child` (dict, Composer if-child node)
- **Returns:** list of `{"lhs": dict, "rhs": dict, "comparator": str}` dicts
- **What it does:** Normalizes three Composer condition shapes into a flat list of atomic conditions:
  1. **Direct shape**: node has `lhs-fn` and `comparator` directly
  2. **Compound**: `condition.condition-type == "compound"` — walks inner `conditions` list recursively via internal `_walk()`
  3. **Binary-compound / binary**: has `lhs`, `rhs`, `comparator`, and optional `tickers` list. Expands one atomic condition per ticker.
- **Drops silently:** nodes that produce no `comparator` after walking (filtered at end)

#### `extract_conditions_from_tree(node, depth, path_conditions, sub_strategy, results, _stats)` — line ~169
- **Parameters:** `node` (dict), `depth` (int, 0 at root), `path_conditions` (list[str], conditions on current path), `sub_strategy` (str|None), `results` (list, accumulator), `_stats` (dict with visited/extracted/skipped counters)
- **Returns:** flat list of condition dicts (see Data Structures section)
- **What it does:** Recursive tree walker. Dispatches on `node["step"]`:
  - `"group"`: updates `sub_strategy` from node name, recurses into children
  - `"root"`, `"wt-cash-equal"`, `"wt-cash-specified"`: recurses into children
  - `"asset"`: returns immediately (leaf)
  - `"filter"`: if `select-n == 1` and exactly 2 asset children and `sort-by-fn` is set, expands into pairwise comparison conditions (winner vs loser for each asset). Otherwise logs skip reason and recurses into children.
  - `"if"`: separates positive and else branches; for each positive branch extracts atomic conditions, builds human label, categorizes, appends to results, then recurses into if-branch children and (with negated label) else-branch children
  - `"if-child"`: recurses into children (does not emit conditions itself)
- **Side effects:** modifies `results` and `_stats` in place; optionally prints debug output if `EXTRACTION_DEBUG = True`

#### `extract_conditions(node)` — line ~322
- **Parameters:** `node` (dict, root of strategy JSON)
- **Returns:** list of condition dicts
- **What it does:** Public wrapper. Initializes stats dict, calls `extract_conditions_from_tree`, prints extraction summary, returns results.

#### `_categorize_condition(lhs, rhs)` — line ~340
- **Parameters:** `lhs` (dict), `rhs` (dict)
- **Returns:** str — category name
- **What it does:** Determines sweep category from lhs/rhs types and fn_labels:
  - If lhs is not indicator: `"unknown"`
  - If rhs is fixed: `"{fn}_fixed"` (e.g. `"RSI_fixed"`)
  - If rhs is indicator: handles special cases Price_vs_EMA, EMA_vs_MA, EMA_vs_EMA; otherwise `"{lhs_fn}_vs_{rhs_fn}"`

#### `_collect_endpoints(node)` — line ~362
- **Parameters:** `node` (dict, if-child JSON node)
- **Returns:** list of str (tickers), order-preserving, deduplicated via `seen` set
- **What it does:** DFS walk of the node's subtree; collects ticker from every `step == "asset"` leaf.

#### `prompt(label, default, cast)` — line ~382
- **Parameters:** `label` (str), `default` (any|None), `cast` (callable, default str)
- **Returns:** value cast to `cast` type
- **What it does:** Interactive CLI prompt with default display and re-prompt on invalid cast. Loops until valid input.

#### `gather_inputs()` — line ~394
- **Parameters:** None (reads `sys.argv`)
- **Returns:** dict with keys: `json_path`, `primary_asset`, `fuzz_pct` (dict), `thresh_step`, `period_step`, `start_date`, `end_date`
- **What it does:** Detects interactive vs non-interactive mode from `sys.argv[1]`. In non-interactive mode, uses all hardcoded defaults. In interactive mode, calls `prompt()` for each parameter. Validates JSON file existence, exits on failure.
- **Side effects:** `sys.exit(1)` if JSON path doesn't exist

#### `load_price_series(ticker)` — line ~456
- **Parameters:** `ticker` (str)
- **Returns:** pd.Series of close prices, indexed by date (datetime)
- **What it does:** Reads CSV via `load_ticker_csv`, sets date index, extracts `close` column. Caches result in module-level `_PRICE_CACHE` dict (key: uppercase ticker) to prevent repeated I/O.
- **Side effects:** populates `_PRICE_CACHE`

#### `compute_indicator(series, fn_label, period)` — line ~466
- **Parameters:** `series` (pd.Series), `fn_label` (str), `period` (int)
- **Returns:** pd.Series of indicator values
- **What it does:** Dispatches to the correct indicator function by fn_label:
  - `"RSI"` → `calculate_rsi`
  - `"MA"` or `"SMA"` → `calculate_sma`
  - `"EMA"` → `calculate_ema`
  - `"CumRet"` → `calculate_cumret`
  - `"MaxDD"` → `calculate_maxdd`
  - `"MAReturn"` → `calculate_mareturn`
  - else → returns `series` unchanged (Price fallback)

#### `get_bil_daily_returns(start_date, end_date)` — line ~483
- **Parameters:** `start_date` (str, YYYY-MM-DD), `end_date` (str)
- **Returns:** pd.Series of BIL daily pct_change, date-filtered, NaN filled to 0
- **Side effects:** triggers `load_price_series("BIL")`

#### `get_primary_daily_returns(ticker, start_date, end_date)` — line ~490
- **Parameters:** `ticker` (str), `start_date` (str), `end_date` (str)
- **Returns:** pd.Series of daily pct_change, date-filtered, NaN filled to 0

#### `_apply_comparator(comp, lhs_vals, rhs_vals)` — line ~505
- **Parameters:** `comp` (str: "gt"/"lt"/"gte"/"lte"/"eq"/"neq"), `lhs_vals`, `rhs_vals` (Series or scalar)
- **Returns:** boolean Series or scalar
- **What it does:** Applies the named comparator. Falls back to `>` for unknown comparator strings.

#### `_evaluate_signal(combined, fired_mask, bil_returns, all_returns, period_val, param_val)` — line ~515
- **Parameters:** `combined` (DataFrame), `fired_mask` (boolean Series), `bil_returns` (Series), `all_returns` (dict ticker→Series), `period_val` (any), `param_val` (any)
- **Returns:** dict with sweep metrics, or None if fewer than 2 signal days or no valid returns
- **What it does:** Computes all sweep-point metrics for signal days:
  - Aligns endpoint next-day returns (`.pct_change().shift(-1)`) to fired days
  - Computes win rate vs BIL
  - Computes log-weighted score: `win_rate * log(total)`
  - Computes profit factor: sum(positive returns) / sum(abs(negative returns)), capped at 99
  - Computes beat_rates: per-ticker win rate vs fired endpoint returns for every ticker in `all_returns`
- **Returns dict keys:** `period`, `param`, `win_rate`, `total_trades`, `score`, `profit_factor`, `beat_rates` (dict)

#### `sweep_condition(cond, config, bil_returns, all_returns, endpoint=None)` — line ~556
- **Parameters:** `cond` (condition dict), `config` (config dict), `bil_returns` (Series), `all_returns` (dict), `endpoint` (str|None)
- **Returns:** `(pd.DataFrame, None)` on success or `(None, error_str)` on failure
- **What it does:** Runs a parameter sweep grid for one condition × one endpoint. Dispatches by category (see Sweep Engine section below). Returns a DataFrame with one row per (period, param) grid point.
- **Side effects:** loads price series, triggers indicator calculations

#### `compute_fragility(df)` — line ~711
- **Parameters:** `df` (pd.DataFrame or None)
- **Returns:** float 0.0–1.0
- **What it does:** Coefficient of variation of `win_rate` across all sweep points: `std(win_rate) / mean(win_rate)`, capped at 1.0. Returns 1.0 if df is None, empty, or mean win rate is 0.

#### `compute_tail_metrics(fired_returns, bil_returns)` — line ~728
- **Parameters:** `fired_returns` (pd.Series of endpoint next-day returns on signal days), `bil_returns` (pd.Series)
- **Returns:** dict with 6 keys (see below)
- **What it does:** Computes tail dependency metrics at base parameters:
  - **base_win_rate**: fraction of signal days where endpoint beats BIL
  - **stripped_win_rate**: same but after removing top 5% days by absolute return magnitude
  - **wr_delta**: `max(base_win_rate - stripped_win_rate, 0)` — WR lost by removing outliers
  - **tail_concentration**: fraction of total gains from top 5% gain days (by return magnitude)
  - **excess_kurtosis**: `mean(((x - mean)/std)^4) - 3` using numpy
  - **tail_score**: `min(0.5 * tail_concentration + 0.5 * min(wr_delta / 0.15, 1.0), 1.0)`
- Returns all-zeros dict if `fired_returns` is None or has fewer than 5 rows.

#### `_get_base_fired_returns(cond, config, bil_returns, all_returns, endpoint)` — line ~789
- **Parameters:** `cond` (condition dict), `config` (config dict), `bil_returns`, `all_returns`, `endpoint` (str)
- **Returns:** pd.Series of endpoint next-day returns on signal-fired days (base params only), or None
- **What it does:** Mirrors the signal-generation logic of `sweep_condition` but only for base parameter values. Used to compute tail metrics without modifying the main sweep loop. Handles all condition categories that `sweep_condition` handles.

#### `fragility_color(score)` — line ~899
- **Parameters:** `score` (float 0–1)
- **Returns:** hex color string
- **What it does:** Linear interpolation across 5 breakpoints: 0→green, 0.25→yellow, 0.5→orange, 0.75→red, 1.0→purple. Picks nearest color at each half-interval (not true linear blend of RGB).

#### `fragility_label(score)` — line ~911
- **Parameters:** `score` (float)
- **Returns:** str — "Robust" / "Stable" / "Moderate" / "Fragile" / "Very Fragile"
- **Thresholds:** < 0.15, < 0.35, < 0.55, < 0.75, else

#### `df_to_heatmap_data(df, cond)` — line ~920
- **Parameters:** `df` (pd.DataFrame or None/str), `cond` (condition dict, unused currently)
- **Returns:** dict with heatmap structure, or None
- **What it does:** Converts sweep DataFrame into JS-ready JSON structure. Unique-sorts periods and params; builds a 2D matrix of cell dicts. Each cell: `{"wr": float, "n": int, "s": float, "pf": float, "pb": dict}`. Sets `is_1d: true` if only one unique param value.

#### `_normalize_prices(prices)` — line ~954
- **Parameters:** `prices` (list of float|None)
- **Returns:** list of float|None, rebased so first non-None value = 100
- **What it does:** Finds first non-None price, divides all values by it and multiplies by 100.

#### `build_signal_data(conditions, config)` — line ~962
- **Parameters:** `conditions` (list of condition dicts), `config` (config dict)
- **Returns:** dict with keys `dates`, `prices`, `returns`, `signals` (see Data Structures)
- **What it does:** Pre-computes all data needed for browser-side rendering of equity curve and signal overlay charts. Collects all tickers, builds a unified date spine (union of all trading dates), forward-fills prices, computes cumulative equity curves for strategy/asset/BIL, and pre-computes signal boolean arrays at base parameters. Keyed per `{cond_id}:{alloc_ticker}`.

#### `generate_html(conditions, sweep_results, reliability_scores, tail_detail, config, signal_data=None)` — line ~1123
- **Parameters:** `conditions` (list), `sweep_results` (dict (cond_id, alloc)→df|str), `reliability_scores` (dict cond_id→{fragility, tail_score, combined}), `tail_detail` (dict cond_id→tail metrics dict), `config` (dict), `signal_data` (dict|None)
- **Returns:** str — full HTML content
- **What it does:** Loads `report_template.html`, serializes all data to JSON, performs string replacements for all `__PLACEHOLDER__` tokens, inlines uPlot JS and CSS (from local cache files or CDN download). Returns error HTML string if template file not found.
- **Side effects:** May download uPlot from CDN and write `uplot.min.js` and `uplot.min.css` to disk if not cached.

#### `main()` — line ~1232
- **Parameters:** None
- **Returns:** None
- **What it does:** Full orchestration:
  1. Clears `_PRICE_CACHE`
  2. Calls `gather_inputs()`
  3. Loads API keys via `load_config()`
  4. Parses strategy JSON
  5. Calls `extract_conditions()` and exits if none found
  6. Collects all tickers (BIL + primary + all lhs/rhs/endpoint tickers)
  7. Calls `check_freshness_and_update()` to download/refresh price CSVs
  8. Loads all daily returns into `all_returns` dict
  9. For each condition × allocation: calls `sweep_condition()`, `compute_fragility()`, and `_get_base_fired_returns()` + `compute_tail_metrics()`
  10. Computes combined score: `0.6 * fragility + 0.4 * tail_score`
  11. Calls `build_signal_data()`
  12. Calls `generate_html()` and writes output file
- **Side effects:** creates HTML file; uses `sys.exit(1)` on no conditions found

---

### `strategy_engine/src/config_loader.py`

#### `load_config(config_filename=None, config_dict=None)` — line ~6
- **Parameters:** `config_filename` (unused), `config_dict` (dict|None, returned as-is)
- **Returns:** tuple `(config_dict or {}, api_keys)` where `api_keys` is list of str
- **What it does:** Loads `.env` from `strategy_engine/.env`, reads `TIINGO_API_KEYS` env var. Parses it as JSON array if it starts with `[`, otherwise splits on commas. Raises `ValueError` if keys missing or empty.
- **Reads:** `strategy_engine/.env`

---

### `strategy_engine/src/data_loader.py`

#### `get_latest_tiingo_date(api_keys)` — line ~13
- **Parameters:** `api_keys` (list of str)
- **Returns:** str — most recent trading date on Tiingo in `YYYY-MM-DD` format
- **What it does:** Fetches SPY prices for last 10 days from Tiingo; takes the last entry's date. Rotates through all API keys on failure.
- **Raises:** Exception if all keys fail

#### `download_ticker_data(ticker, api_keys, data_dir)` — line ~38
- **Parameters:** `ticker` (str), `api_keys` (list of str), `data_dir` (Path)
- **Returns:** True on success
- **What it does:** Downloads full historical daily data from Tiingo (startDate=1900-01-01), rotates keys on failure. Implements exponential backoff for 429 responses (up to 5 retries, starting at 1s, capped at 16s). Saves as CSV with columns `date`, `close` (adjClose). Sanitizes ticker with `/` → `-` and `.` → `-` for filename.
- **Raises:** Exception if all keys exhausted
- **Side effects:** creates `data_dir/{safe_ticker}.csv`

#### `check_freshness_and_update(tickers, api_keys, data_dir)` — line ~86
- **Parameters:** `tickers` (list of str), `api_keys` (list of str), `data_dir` (Path)
- **Returns:** None
- **What it does:** For each ticker: checks if CSV exists and its max date >= latest market date. If stale or missing, calls `download_ticker_data()` followed by `time.sleep(0.4)` pacing delay.
- **Side effects:** downloads/writes CSVs; prints status for each ticker

---

### `strategy_engine/src/data_alignment.py`

#### `load_ticker_csv(ticker, data_dir)` — line ~4
- **Parameters:** `ticker` (str), `data_dir` (Path)
- **Returns:** pd.DataFrame with columns `date` (datetime), `close` (float), sorted ascending
- **What it does:** Reads `{data_dir}/{safe_ticker}.csv`, parses date, sorts. Sanitizes ticker with `/` → `-` and `.` → `-`.
- **Raises:** FileNotFoundError if CSV doesn't exist

---

### `strategy_engine/src/indicators.py`

#### `calculate_sma(series, period)` — line ~4
- **Parameters:** `series` (pd.Series), `period` (int)
- **Returns:** pd.Series — `series.rolling(window=period).mean()`

#### `calculate_ema(series, period)` — line ~8
- **Parameters:** `series` (pd.Series), `period` (int)
- **Returns:** pd.Series — `series.ewm(span=period, adjust=False).mean()`

#### `calculate_rsi(series, period)` — line ~12
- **Parameters:** `series` (pd.Series), `period` (int)
- **Returns:** pd.Series of RSI values 0–100
- **What it does:** Wilder's smoothing RSI using EWM with `alpha=1/period`. First `period` values set to NaN. Inf values (zero avg_loss) replaced with 100.

#### `calculate_cumret(series, period)` — line ~30
- **Parameters:** `series` (pd.Series), `period` (int)
- **Returns:** pd.Series — `series.pct_change(periods=period) * 100`
- **Note:** Returns percentage (not decimal), e.g. 5.0 means 5%.

---

## Condition Extractor

### `extract_conditions_from_tree()` — Complete Detail

**JSON Node Shapes Handled:**

| `step` value | Action |
|---|---|
| `"group"` | Updates `sub_strategy` from node name; recurses children |
| `"root"`, `"wt-cash-equal"`, `"wt-cash-specified"` | Recurses children unchanged |
| `"asset"` | Returns immediately — leaf node, no condition emitted |
| `"filter"` | Special handling for select-1-of-2 pairwise filter (see below) |
| `"if"` | Main extraction site — separates positive/else branches |
| `"if-child"` | Container node; recurses children without extracting |
| anything else | No action (falls through to `return results`) |

**Filter step special case:** Only expands into conditions when `select-n == 1`, exactly 2 asset children, and `sort-by-fn` is non-empty. Emits one condition per asset (winner vs loser), with `children_endpoints: [winner_ticker]`. Portfolio filters (select-n != 1 or more than 2 assets) are logged as skipped and their children are still walked.

**Condition payload shapes (inside `_extract_atomic_conditions`):**

1. **Direct if-child**: node has `lhs-fn` and `comparator` at top level → `_parse_side()` for each side
2. **Compound**: `condition.condition-type == "compound"` → recursively walks inner `conditions` list; no ticker inherited at this level
3. **Binary-compound** or **binary**: has `comparator`, `lhs` (dict with `fn`/`params`/`ticker`), `rhs` (dict or `constant`), and optional `tickers` list. Expands to one atomic condition per ticker. If `tickers` is empty and `inherited_ticker` is available, uses that. `%` ticker placeholder is replaced by ticker_override.

**Silent drops:**
- Nodes with no `comparator` after walking (filtered by `[x for x in out if x.get("comparator")]`)
- `if` nodes whose positive branches yield no atomic conditions (`if not atomic_conditions: continue`)
- `filter` nodes that don't match the select-1-of-2 pattern (logged as skipped but subtree is still walked)
- JSON nodes with `step` not in the handled set (no conditions emitted, no error)
- `_parse_side_from_condition_spec` called with a non-dict returns a blank indicator spec (no crash)

**Output condition dict schema (every field):**

```python
{
    "id":                  int,      # sequential index, 0-based
    "sub_strategy":        str,      # nearest ancestor group name, or "(root)"
    "depth":               int,      # nesting depth (increments per positive if-branch)
    "human":               str,      # e.g. "RSI(QQQ, 10) > 79.0"
    "comparator":          str,      # "gt", "lt", "gte", "lte", "eq", "neq"
    "comp_label":          str,      # ">", "<", ">=", "<=", "==", "!="
    "lhs":                 dict,     # side dict (see below)
    "rhs":                 dict,     # side dict (see below)
    "category":            str,      # see category table
    "path_so_far":         list[str],# human labels of ancestor conditions
    "children_endpoints":  list[str],# asset tickers reachable if condition fires
}
```

**Side dict schema:**
```python
# Fixed value side:
{"type": "fixed", "value": float}

# Indicator side:
{
    "type":     "indicator",
    "fn_raw":   str,    # raw Composer function name, e.g. "relative-strength-index"
    "fn_label": str,    # normalized label: "RSI", "MA", "EMA", "CumRet", "MaxDD", "MAReturn", "Price"
    "ticker":   str,    # e.g. "QQQ", or "?" if not found
    "window":   int|None
}
```

---

## Sweep Engine

### `sweep_condition()` — Complete Detail

**Category routing and sweep axes:**

**Category 1: RSI_fixed, CumRet_fixed, MaxDD_fixed, MAReturn_fixed** (2D sweep)
- Axis 1 (period): `range(max(2, round(base_period*(1-f))), max(3, round(base_period*(1+f)))+1, period_step)`
- Axis 2 (threshold): `np.arange(t_lo, t_hi + thresh_step/2, thresh_step)` around base threshold ±f%
- DataFrame column `period` = period, `param` = threshold value
- Indicator computed once per period, applied across all thresholds
- CumRet note: base threshold is the raw `rhs["value"]` which may be a percentage (not decimal)

**Category 2: RSI_vs_RSI, CumRet_vs_CumRet, MaxDD_vs_MaxDD, MAReturn_vs_MAReturn, MaxDD_vs_MAReturn, MAReturn_vs_MaxDD, MA_vs_MA, EMA_vs_EMA** (2D sweep, independent left/right windows)
- Axis 1 (lhs period): range around `lhs["window"]` ±f%
- Axis 2 (rhs period): range around `rhs["window"]` ±f%
- DataFrame column `period` = lhs_period, `param` = rhs_period
- Loads price_l from lhs ticker, price_r from rhs ticker separately
- fuzz key extracted from category via `cat.split('_')[0]` — for `MaxDD_vs_MAReturn` this yields `"MaxDD"` fuzz pct; for `MAReturn_vs_MaxDD` yields `"MAReturn"` fuzz pct (falls back to 0.2 if not in fuzz dict)

**Category 3: Price_vs_MAReturn** (1D sweep on rhs window)
- Sweeps MAReturn window only
- DataFrame column `period` = window, `param` = `"win_rate"` (literal string — not a numeric param)
- lhs is raw price series; rhs is MAReturn indicator

**Category 4: Price_vs_MA, Price_vs_EMA, MA_fixed, Price_fixed** (1D sweep on rhs window)
- Sweeps MA/EMA window only
- lhs is raw price, rhs is MA or EMA computed over window
- DataFrame column `period` = window, `param` = `"win_rate"` (literal string)
- MA_fixed and Price_fixed: treated the same as Price_vs_MA — takes `rhs.get("window") or lhs.get("window") or 200` as base

**Category 5: EMA_vs_MA** (2D sweep)
- Independent EMA window (lhs) × MA window (rhs) grid
- Single price series loaded from `lhs["ticker"]`
- EMA computed with `compute_indicator(price, "EMA", ema_w)`
- MA computed with `compute_indicator(price, "SMA", ma_w)` — uses SMA function regardless of how MA is labeled

**Unsupported category:** returns `(None, "Unsupported category: {cat}")`

**Metrics computed per sweep point (by `_evaluate_signal`):**
- `win_rate`: fraction of signal days where endpoint next-day return > BIL return
- `total_trades`: count of valid signal days with next-day return available
- `score`: `win_rate * log(max(total_trades, 1))` — log-weighted win rate
- `profit_factor`: `sum(positive returns) / sum(abs(negative returns))`, default 2.0 if no losses, 0.0 if no gains and no losses, capped at 99.0
- `beat_rates`: dict mapping every ticker in `all_returns` to fraction of signal days where endpoint beat that ticker

**Merge conflict detail:**
There is a known merge conflict in the `sweep_condition` function header comment block (lines ~501–503). The code contains a duplicated comment block:
```
# ---------------------------------------------------------------------------
# Single condition sweep
# ---------------------------------------------------------------------------
```
appearing twice in a row (lines 501–503 and 502–504). This is a cosmetic artifact of a poorly resolved git merge — the function body itself is present only once and is not duplicated. The function implementation is correct. No variable names are in conflict; the duplication is comment-only.

---

## HTML Report

### Template System
The report is generated by loading `report_template.html` and doing string `.replace()` for each `__PLACEHOLDER__` token. No f-strings; no Jinja. JS uses `{{}}` literals (previously an f-string artifact).

### Python-side Injections

| Placeholder | Python expression | Type injected |
|---|---|---|
| `__JSON_NAME__` | `Path(config["json_path"]).name` | str |
| `__TIMESTAMP__` | `datetime.now().strftime("%Y-%m-%d %H:%M")` | str |
| `__COND_COUNT__` | `str(len(conditions))` | str |
| `__PRIMARY_ASSET__` | `config["primary_asset"]` | str |
| `__HEATMAP_JSON__` | `json.dumps(heatmap_data)` | JSON |
| `__RELIABILITY_JSON__` | `json.dumps({str(k): v ...})` | JSON |
| `__TAIL_METRICS_JSON__` | `json.dumps({str(k): v ...})` | JSON |
| `__CONDITIONS_JSON__` | `json.dumps(conds_for_js)` | JSON |
| `__AVAILABLE_ASSETS__` | `json.dumps(sorted(all_returns.keys()))` | JSON |
| `__SIGNAL_DATA__` | `json.dumps(signal_data)` | JSON |
| `__UPLOT_JS__` | uPlot 1.6.31 IIFE minified JS | raw JS |
| `__UPLOT_CSS__` | uPlot 1.6.31 CSS | raw CSS |

### `__SIGNAL_DATA__` Structure

```javascript
{
  dates: [str, ...],          // YYYY-MM-DD strings, unified date spine
  prices: {                   // ticker → [float|null, ...]  raw close prices, forward-filled
    "QQQ": [...],
    "BIL": [...],
    ...
  },
  returns: {                  // ticker → [float, ...]  daily pct_change, forward-filled
    "QQQ": [...],
    ...
  },
  signals: {                  // key = "{condId}:{allocTicker}"
    "0:TQQQ": {
      price:       [float|null,...],  // endpoint raw close prices (forward-filled)
      lhs_vals:    [float|null,...],  // lhs indicator at base params
      signal:      [bool,...],        // pre-computed signal mask
      strat_curve: [float,...],       // strategy equity curve, starts at 100
      asset_curve: [float,...],       // buy-and-hold equity curve, starts at 100
      bil_curve:   [float,...],       // BIL equity curve, starts at 100
      label:       str,               // e.g. "RSI(QQQ, 10)"
      human:       str,               // full condition label
      ep:          [float|null,...],  // endpoint price normalized to 100
      lhs:         [float|null,...],  // lhs raw prices
      rhs:         [float|null,...],  // rhs raw prices (null if fixed)
      fn_l:        str|null,          // lhs fn_label
      fn_r:        str|null,          // rhs fn_label (null if fixed)
      wl:          int|null,          // lhs window
      wr:          int|null,          // rhs window (null if fixed)
      fv:          float|null,        // rhs fixed value (null if indicator)
      cmp:         str,               // comparator string e.g. "gt"
      bp:          int|null,          // base param (= lhs window)
      bp2:         int|float|null,    // base param 2 (= rhs window or fixed value)
      ep_t:        str,               // endpoint ticker string
    },
    ...
  }
}
```

### Fragility Score Formula
`fragility = min(std(win_rate) / mean(win_rate), 1.0)` (coefficient of variation, capped at 1.0)

### Combined Score Formula
`combined = 0.6 * fragility + 0.4 * tail_score`

### Charts Generated (4 per condition × allocation)

1. **Equity Curves** (`#eq-c`): uPlot line chart on log scale. Three series: Strategy (enters endpoint on signal, BIL otherwise), Asset buy-and-hold, BIL. Signal periods shaded as background bands. Y-axis: indexed to 100.

2. **Price & Signal Overlay** (`#so-c`): uPlot line chart. Endpoint price normalized to 100. Signal-fired periods highlighted. Shows when in time the condition is active.

3. **Indicator Values** (`#ind-c`): uPlot line chart. LHS indicator value over time. Threshold line drawn if fixed. Helps visualize indicator behavior relative to threshold.

4. **Profile** (`#profile-c`): uPlot chart with win rate (left axis) and profit factor (right axis) as bar-style or line series. Data sourced from sweep DataFrame at selected cell (period, param).

### Heatmap Structure (per condition × allocation)
```javascript
{
  periods: [str, ...],    // sorted unique period values
  params:  [str, ...],    // sorted unique param values
  matrix:  [[cell, ...]], // [periods][params] — null for missing grid points
  is_1d:   bool           // true if only one unique param value (1D sweep categories)
}
// cell object:
{
  wr: float,   // win_rate
  n:  int,     // total_trades
  s:  float,   // score
  pf: float,   // profit_factor
  pb: {}       // beat_rates dict: ticker → float
}
```

---

## Data Structures

### Condition Dict (complete schema)
See Condition Extractor section above.

### Sweep Result DataFrame
One row per (period, param) grid point. Columns:
- `period`: int or str — lhs window/period
- `param`: int, float, or str ("win_rate" for 1D categories) — rhs window/threshold
- `win_rate`: float 0–1
- `total_trades`: int
- `score`: float (win_rate × log(total_trades))
- `profit_factor`: float (capped at 99)
- `beat_rates`: dict (not a proper DataFrame column — stored as object per cell)

### Reliability Scores Dict
Keyed by `cond_id` (int):
```python
{
    "fragility":  float,  # CV of win_rate across sweep, 0–1
    "tail_score": float,  # 0–1, blend of tail_concentration and wr_delta
    "combined":   float   # 0.6*fragility + 0.4*tail_score
}
```

### Tail Detail Dict
Keyed by `cond_id` (int):
```python
{
    "tail_score":         float,  # 0–1 composite tail dependency score
    "tail_concentration": float,  # fraction of gains from top 5% gain days
    "excess_kurtosis":    float,  # numpy kurtosis - 3
    "base_win_rate":      float,  # win rate at base params vs BIL
    "stripped_win_rate":  float,  # win rate with top 5% magnitude days removed
    "wr_delta":           float,  # base_win_rate - stripped_win_rate (floored at 0)
}
```

### Config Dict
```python
{
    "json_path":      Path,
    "primary_asset":  str,    # e.g. "TQQQ"
    "fuzz_pct": {
        "RSI":    float,      # e.g. 0.30
        "MA":     float,
        "CumRet": float,
        "Price":  float,
        "MaxDD":  float,
    },
    "thresh_step":  float,    # e.g. 0.5
    "period_step":  int,      # e.g. 1
    "start_date":   str,      # "YYYY-MM-DD"
    "end_date":     str,
    # Added by main() before generate_html():
    "available_assets": list[str],  # sorted keys of all_returns
}
```

---

## External Dependencies

| Package | Used for |
|---|---|
| `pandas` | DataFrame construction, date indexing, rolling calculations, pct_change |
| `numpy` | `np.maximum.accumulate` for MaxDD, `np.arange` for threshold grids, kurtosis calculation, NaN handling |
| `requests` | Tiingo API calls in `data_loader.py`; uPlot CDN download in `generate_html()` |
| `python-dotenv` | Loading `strategy_engine/.env` in `config_loader.py` |
| `scipy` | Listed in requirements; NOT currently imported anywhere in the codebase (comment in `compute_tail_metrics` says "no scipy needed") |

**Standard library:** `json`, `sys`, `math`, `os`, `time`, `pathlib.Path`, `datetime`

---

## File I/O

### Read
- `config["json_path"]` — Composer strategy export JSON (user-specified)
- `strategy_engine/.env` — Tiingo API keys
- `strategy_engine/data/{safe_ticker}.csv` — price CSVs (one per ticker)
- `report_template.html` — HTML report template
- `uplot.min.js` — uPlot library (local cache)
- `uplot.min.css` — uPlot CSS (local cache)

### Write
- `strategy_engine/data/{safe_ticker}.csv` — price CSVs downloaded from Tiingo
- `uplot.min.js` — cached on first CDN download
- `uplot.min.css` — cached on first CDN download
- `fuzz_report_{json_stem}_{YYYYMMDD_HHMMSS}.html` — output report

---

## Known Issues

### 1. Merge conflict artifact (cosmetic, not a bug)
The comment block `# Single condition sweep` and its surrounding dashes appears twice in a row at lines ~501–504. This is a leftover from a poorly resolved git merge. The actual function implementation is not duplicated and is correct. No code is broken; it is a cosmetic issue only. The variable names referenced in HANDOFF.md's "merge conflict" warning appear to refer to this duplication. No actual variable-name conflict exists in the current code.

### 2. `param` column as literal string "win_rate" for 1D categories
For `Price_vs_MA`, `Price_vs_EMA`, `MA_fixed`, `Price_fixed`, and `Price_vs_MAReturn`, the `param` column in the sweep DataFrame is set to the string `"win_rate"` rather than a numeric value. This means the heatmap will have a single-column matrix (is_1d = true). The JS handles this correctly but the name is confusing — it does not mean the column contains win rate values, it is simply a placeholder label.

### 3. MAReturn units vs CumRet units inconsistency
`calculate_cumret` returns percentage (multiplies by 100), but `calculate_mareturn` returns decimal (no multiplication). If a user strategy has a CumRet condition with threshold like `5.0` (meaning 5%), that threshold is correctly compared to the CumRet series (which is also in %). But a `MAReturn_fixed` threshold would be compared to a decimal series (e.g. 0.001). The condition threshold from the Composer JSON must be in matching units — which is correct for MAReturn (Composer stores it as decimal) but requires awareness when debugging.

### 4. `scipy` listed as dependency but not used
`requirements.txt` (implied) includes scipy, but the codebase uses only numpy for kurtosis. The `compute_tail_metrics` docstring explicitly notes "no scipy needed."

### 5. Short group names hidden in JS sidebar
The `buildTree()` function in `report_template.html` silently skips groups whose names are ≤ 2 characters. Composer exports sometimes assign numeric IDs ("1", "2") as group names. Conditions in those groups will still be swept and present in the data but will not appear as named groups in the sidebar tree.

### 6. Fuzz key extraction for cross-family categories
For `MaxDD_vs_MAReturn`, the fuzz percentage is looked up by `cat.split('_')[0]` = `"MaxDD"`. For `MAReturn_vs_MaxDD` it's `"MAReturn"`. Since neither of these keys exists in the `fuzz_pct` dict (which only has RSI, MA, CumRet, Price, MaxDD), `MAReturn` will fall back to `0.2` default. This is a minor inconsistency; `MaxDD` fuzz pct is used for MaxDD-on-left categories but not for MAReturn-on-left ones.

### 7. `_get_base_fired_returns` duplicates `sweep_condition` logic
All the signal-generation code is written twice: once in `sweep_condition` (for the full sweep grid) and once in `_get_base_fired_returns` (for base params only). This is a maintenance surface — changes to signal generation must be applied in both places.
