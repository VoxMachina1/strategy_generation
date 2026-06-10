# monte_carlo_sim — Functionality Inventory

**File inventoried:** `C:\Python Projects\signal_pipeline\monte_carlo_sim\Monte Carlo walk forward composer working.py`

---

## Overview

Single-file Monte Carlo simulation tool for evaluating forward return risk of Composer.Trade algorithmic strategies. It fetches a strategy's full backtest data via the Composer public API, reconstructs daily portfolio returns by downloading price data from Yahoo Finance, then runs bootstrap Monte Carlo simulations (sampling historical returns to generate 10,000 simulated equity paths). It validates the strategy's historical performance against its own simulated forward distribution using walk-forward testing: how often did the actual result fall within the model's confidence intervals?

Supports two testing modes: a standard walk-forward test (train on all data except the last N days, evaluate the last N days) and a rolling walk-forward test (slide a fixed training window forward in steps, generating many out-of-sample test periods).

---

## Entry Points

### `main()` — line 1746

**Command line:** `python "Monte Carlo walk forward composer working.py"` (no args)

**Interactive prompts:**
1. `Enter Composer Symphony URL (default: ...)` — accepts a Composer symphony URL (extracts the ID from the path)
2. `Enter test mode (1, 2, or 3, default: 1)` — selects Standard Walk-Forward, Rolling Walk, or both
3. If mode 2 or 3:
   - `Enter training period length in days (default: 504)`
   - `Enter test period length in days (default: 252)`
   - `Enter step size in days (default: 252)`

**What it produces:**
- `composer_monte_carlo_results/` directory containing:
  - `{name}_daily_returns.csv` — historical daily returns CSV
  - `{name}_walk_forward_{N}d.png` — Monte Carlo fan chart overlaid with actual path, per period length (63, 126, 252, 504 days)
  - `{name}_drawdown_analysis_{N}d.png` — 6-panel drawdown analysis chart per period
  - `{name}_drawdown_distributions_{N}d.png` — side-by-side histogram of simulated vs actual max drawdown and duration (only for periods >= 63 days)
  - `{name}_cagr_distribution_{N}d.png` — CAGR histogram (only for periods >= 252 days)
  - `{name}_comparison.png` — bar chart of actual vs forecast return per period length
  - `{name}_walk_forward_results.csv` — summary table of all walk-forward results
  - If rolling mode: `{name}_rolling_walk/` subdirectory containing per-iteration charts, rolling comparison charts, and `{name}_rolling_results.csv`

---

## Functions & Classes (exhaustive)

### `convert_trading_date(date_int)` — line 30
- **Parameters:** `date_int` (int) — number of days since 1970-01-01
- **What it does:** Converts a Composer API integer trading date to a Python `datetime` object
- **Returns:** `datetime`
- **Side effects:** None

---

### class `YahooFinanceAPI` — line 39

**`__init__(self, session=None)`** — line 42
- Initializes the yfinance wrapper. Sets `ticker_map` (`{'BRK/B': 'BRK-B'}`), `rate_limit_delay=1.0`, `use_batch_download=True`, `batch_size=5`.
- **Side effects:** Prints initialization message

**`fetch_historical_data(self, symbols, start_date, end_date)`** — line 68
- **Parameters:** `symbols` (List[str]), `start_date` (str YYYY-MM-DD), `end_date` (str YYYY-MM-DD)
- **What it does:** Routes to batch or individual download based on `use_batch_download` and symbol count. Applies ticker mapping before dispatching.
- **Returns:** `Dict[str, pd.Series]` — maps original symbol name to adjusted Close price series (daily, timezone-naive, float32, no duplicates)
- **Side effects:** Prints progress messages

**`_individual_download(self, mapped_symbols, start_date, end_date)`** — line 99
- **Parameters:** `mapped_symbols` (Dict[yahoo_symbol -> original_symbol]), date strings
- **What it does:** Downloads each ticker one at a time via `yfinance.Ticker.history()`. Cleans NaNs, casts to float32, strips timezone, deduplicates. Falls back to `{TICKER}-USD` format on failure. Sleeps `rate_limit_delay` between each ticker.
- **Returns:** `Dict[str, pd.Series]`
- **Side effects:** Sleeps between requests; prints progress; may print error and try alternate symbol format

**`_batch_download(self, mapped_symbols, start_date, end_date)`** — line 213
- **Parameters:** `mapped_symbols` (Dict), date strings
- **What it does:** Downloads groups of `batch_size` tickers at once via `yfinance.download()`. Processes multi-ticker vs single-ticker response structures differently. Retries any failed symbols individually via `_individual_download`. Sleeps between batches.
- **Returns:** `Dict[str, pd.Series]`
- **Side effects:** Prints progress; sleeps between batches

---

### `fetch_backtest(id, start_date, end_date)` — line 325
- **Parameters:** `id` (str) — Composer symphony URL or ID, `start_date` (str), `end_date` (str)
- **What it does:** Extracts the symphony ID from the URL. POSTs to the Composer backtest API with $100,000 capital, v2 backtest, slippage 0.05%. Parses the returned JSON to extract symphony name, current holdings, and `tdvm_weights` (allocation weights by ticker and trading date integer). Builds a date-range DataFrame with allocation percentages.
- **Returns:** `(allocations_df, symphony_name, tickers)` — DataFrame indexed by date with ticker columns (percent allocations), the symphony display name, and list of ticker strings
- **Side effects:** HTTP POST to `https://backtest-api.composer.trade/api/v2/public/symphonies/{id}/backtest`; prints nothing

---

### `calculate_portfolio_returns(allocations_df, tickers)` — line 369
- **Parameters:** `allocations_df` (pd.DataFrame — allocation percents indexed by date), `tickers` (list of str)
- **What it does:** Strips non-trading days from allocations, normalizes dates, fetches price history via `YahooFinanceAPI`, computes daily price changes per ticker, then computes a weighted daily portfolio return using prior-day allocations (correct, no lookahead). Forward-fills missing prices. Prints debug info for last 5 days.
- **Returns:** `(daily_returns, dates)` — pd.Series of daily % returns and corresponding date index
- **Side effects:** Prints return statistics and date alignment info; calls `YahooFinanceAPI.fetch_historical_data()`

---

### `run_monte_carlo_simulation(returns, num_simulations, simulation_length, annual_periods)` — line 544
- **Parameters:** `returns` (array-like of daily % returns), `num_simulations` (int, default 10000), `simulation_length` (int, default len(returns)), `annual_periods` (int, default 252)
- **What it does:** Separates historical returns into positive and negative pools. For each simulation, samples returns by drawing from each pool based on empirical probability of a positive day (preserves the positive/negative frequency). Computes compounding cumulative returns, max drawdown, Sharpe ratio, and drawdown duration tracking for each path. Calculates 5th/25th/50th/75th/95th percentile paths.
- **Returns:** dict with keys:
  - `final_returns` — np.array of final cumulative return per simulation
  - `paths` — np.array (num_simulations × simulation_length+1) of cumulative returns
  - `percentiles` — dict with keys `'5'`, `'25'`, `'50'`, `'75'`, `'95'`, each a path array
  - `sharpe_ratios` — np.array
  - `max_drawdowns` — np.array (as fraction, not percent)
  - `max_drawdown_durations` — np.array (in trading days)
  - `total_drawdown_days` — np.array
- **Side effects:** Prints return characteristic summary

---

### `analyze_drawdowns(returns, output_dir, period_length, test_start_date, test_end_date, portfolio_name, dates)` — line 675
- **Parameters:** `returns` (list of cumulative % returns), `output_dir` (str), `period_length` (int), `test_start_date` (str), `test_end_date` (str), `portfolio_name` (str), `dates` (optional list of date strings)
- **What it does:** Walks the cumulative return series to identify all drawdown periods (start/end index, trading-day duration, calendar-day duration, max depth). Sorts by severity. Creates a 6-panel matplotlib figure: underwater plot, cumulative return with drawdown shading, grouped bar chart of episode durations, drawdown distribution histogram, duration distribution histogram, and magnitude-vs-duration scatter with regression line.
- **Returns:** dict with keys: `max_drawdown`, `avg_drawdown`, `total_drawdown_days`, `significant_drawdown_days`, `avg_drawdown_length`, `avg_calendar_days`, `drawdown_periods`, `significant_periods`, `max_drawdown_duration`, `max_calendar_duration`, `drawdown_durations`, `calendar_durations`, `drawdown_magnitudes`, `top_significant_periods`
- **Side effects:** Writes PNG to `{output_dir}/{portfolio_name}_drawdown_analysis_{period_length}d.png`; prints drawdown statistics

---

### `plot_drawdown_distributions(simulation_results, actual_max_drawdown, actual_dd_duration, period_length, output_dir, portfolio_name)` — line 1052
- **Parameters:** simulation results dict (from `run_monte_carlo_simulation`), actual max drawdown (float, percent), actual DD duration (int, calendar days), period length (int), output dir (str), portfolio name (str)
- **What it does:** Creates a 2-panel figure: left panel is histogram of simulated max drawdowns with actual marked and percentile; right panel is histogram of estimated calendar-day drawdown durations (trading days scaled by 1.45) with actual marked and percentile. Includes statistics boxes.
- **Returns:** dict with keys: `dd_mean`, `dd_median`, `dd_std`, `dd_5th`, `dd_95th`, `dd_percentile`, `dur_mean`, `dur_median`, `dur_std`, `dur_5th`, `dur_95th`, `dur_percentile`
- **Side effects:** Writes PNG to `{output_dir}/{portfolio_name}_drawdown_distributions_{period_length}d.png`

---

### `run_walk_forward_test(dates, returns, test_period_length, output_dir, portfolio_name)` — line 1166
- **Parameters:** `dates` (list of date strings), `returns` (list of daily % returns), `test_period_length` (int), `output_dir` (str), `portfolio_name` (str)
- **What it does:** Splits data into train (all but last N days) and test (last N days). Runs 10,000 Monte Carlo simulations on training data with `simulation_length = test_period_length`. Calculates actual cumulative path for the test period. Calls `analyze_drawdowns()` on the actual test path. For periods >= 63 days, calls `plot_drawdown_distributions()`. For periods >= 252 days, generates a CAGR distribution plot. Creates a fan chart comparing simulated percentile bands to actual path. Calculates percentile rank of actual result.
- **Returns:** dict with many keys including `period_length`, `test_start_date`, `test_end_date`, `actual_final_return`, `actual_annualized_return`, `actual_sharpe`, `actual_max_drawdown`, `actual_dd_duration_trading`, `actual_dd_duration_calendar`, `actual_percentile`, `median_forecast`, `forecast_error`, `percent_error`, `in_90_interval`, `in_50_interval`, plus drawdown stats and optionally CAGR/DD distribution stats
- **Side effects:** Writes multiple PNGs; prints detailed results summary

---

### `run_rolling_walk_forward_test(dates, returns, train_period_length, test_period_length, output_dir, portfolio_name, step_size)` — line 1438
- **Parameters:** `dates` (list of str), `returns` (list of floats), `train_period_length` (int), `test_period_length` (int), `output_dir` (str), `portfolio_name` (str), `step_size` (int, defaults to test_period_length)
- **What it does:** Creates `{portfolio_name}_rolling_walk/` subdirectory. Iterates by sliding a training window forward by `step_size` each time. For each iteration, runs Monte Carlo on the training slice and evaluates against the next `test_period_length` days. For each iteration: calls `analyze_drawdowns()` and saves a fan chart PNG. After all iterations, creates: actual vs forecast returns bar chart, CAGR comparison bar chart (if periods >= 20 days), max drawdown bar chart, and saves a summary CSV.
- **Returns:** pd.DataFrame of rolling results, or None if insufficient data
- **Side effects:** Creates subdirectory; writes many PNGs; writes CSV `{name}_rolling_results.csv`; prints summary

---

## Data Structures

### Allocations DataFrame (output of `fetch_backtest`)
- Index: `pd.DatetimeIndex` (full calendar date range)
- Columns: one column per ticker (str), values are allocation percentages (0–100)
- Special column: `$USD` (cash position)

### Prices DataFrame (inside `calculate_portfolio_returns`)
- Index: `pd.DatetimeIndex` (trading days, timezone-naive)
- Columns: one column per ticker, values are adjusted close price (float32)
- Special column: `$USD` = 1.0

### Monte Carlo results dict (output of `run_monte_carlo_simulation`)
```python
{
  'final_returns':          np.array of shape (num_simulations,),  # final cumulative %
  'paths':                  np.array of shape (num_simulations, simulation_length+1),
  'percentiles': {
    '5':  np.array, '25': np.array, '50': np.array,
    '75': np.array, '95': np.array                  # shape (simulation_length+1,)
  },
  'sharpe_ratios':          np.array,
  'max_drawdowns':          np.array,   # fractional (not percent)
  'max_drawdown_durations': np.array,   # trading days
  'total_drawdown_days':    np.array,
}
```

### Drawdown period dict (inside `analyze_drawdowns`)
```python
{
  'start_idx':    int,
  'end_idx':      int,
  'start_date':   str,
  'end_date':     str,
  'duration':     int,   # trading days
  'max_drawdown': float, # percent (positive number)
  'calendar_days': int,
}
```

---

## External Dependencies

| Library | Usage |
|---|---|
| `numpy` | Array operations, random sampling, percentile calculations |
| `pandas` | DataFrames for allocations, prices, returns |
| `matplotlib` | All charting (fan charts, bar charts, scatter plots) |
| `seaborn` | Histogram plots with KDE (`histplot`) |
| `scipy.stats` | `percentileofscore()` — ranking actual vs simulated distribution |
| `yfinance` | Historical price download (auto-installed if missing) |
| `requests` | HTTP POST to Composer backtest API |

---

## File I/O

| File | Format | Read/Write | Notes |
|---|---|---|---|
| `composer_monte_carlo_results/{name}_daily_returns.csv` | CSV | Write | Date and Daily_Return columns |
| `composer_monte_carlo_results/{name}_walk_forward_{N}d.png` | PNG | Write | Fan chart per test period |
| `composer_monte_carlo_results/{name}_drawdown_analysis_{N}d.png` | PNG | Write | 6-panel drawdown analysis |
| `composer_monte_carlo_results/{name}_drawdown_distributions_{N}d.png` | PNG | Write | Only for periods >= 63 days |
| `composer_monte_carlo_results/{name}_cagr_distribution_{N}d.png` | PNG | Write | Only for periods >= 252 days |
| `composer_monte_carlo_results/{name}_comparison.png` | PNG | Write | Actual vs forecast bar chart |
| `composer_monte_carlo_results/{name}_walk_forward_results.csv` | CSV | Write | All walk-forward metrics |
| `composer_monte_carlo_results/{name}_rolling_walk/{name}_rolling_iter{N}.png` | PNG | Write | Per rolling iteration |
| `composer_monte_carlo_results/{name}_rolling_walk/{name}_rolling_returns_comparison.png` | PNG | Write | Rolling summary |
| `composer_monte_carlo_results/{name}_rolling_walk/{name}_rolling_cagr_comparison.png` | PNG | Write | Rolling CAGR summary |
| `composer_monte_carlo_results/{name}_rolling_walk/{name}_rolling_drawdowns.png` | PNG | Write | Rolling max drawdowns |
| `composer_monte_carlo_results/{name}_rolling_walk/{name}_rolling_results.csv` | CSV | Write | Rolling results table |

---

## API Calls

| Endpoint | Method | What it fetches |
|---|---|---|
| `https://backtest-api.composer.trade/api/v2/public/symphonies/{id}/backtest` | POST | Full backtest data: legend, last holdings, `tdvm_weights` (allocation weights keyed by ticker and integer trading date) |
| `https://api.tiingo.com/tiingo/daily/SPY/prices` (via yfinance) | — | Adjusted close prices via yfinance for all portfolio tickers |

The Composer API payload: `{"capital": 100000, "apply_reg_fee": true, "apply_taf_fee": true, "backtest_version": "v2", "slippage_percent": 0.0005, "start_date": ..., "end_date": ...}`

---

## Known Issues / Dead Code

- **`drawdown_magnitudes` variable shadowing:** Inside `analyze_drawdowns()`, the local variable `max_drawdowns` (a list of per-period max drawdowns) shadows the parameter name used in the outer context. This is a readability hazard but not a bug.
- **Drawdown is fractional in simulation, percent in actual:** `run_monte_carlo_simulation` stores max_drawdowns as fractions (e.g., 0.15 = 15%), while `analyze_drawdowns` works in percent (e.g., 15.0). The `actual_max_drawdown` passed to `plot_drawdown_distributions` is in percent, while `simulation_results['max_drawdowns']` is in fractions — there is no unit reconciliation at that handoff. This is a **latent bug** in the drawdown distribution comparison chart.
- **`calendar_days` scaling approximation:** Duration histogram for simulated paths multiplies trading days by 1.45 to estimate calendar days (hardcoded, line ~1104). This is an approximation.
- **`first_valid_index` is computed but never used** in `calculate_portfolio_returns` (line 375).
- **No `signal_operator` column in results** — this is a standalone script, not part of the rsi_search pipeline.
