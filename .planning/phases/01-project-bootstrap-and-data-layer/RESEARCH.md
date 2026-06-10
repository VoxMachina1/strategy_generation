# Phase 1: Project Bootstrap and Data Layer — Research

**Researched:** 2026-06-10
**Domain:** Python package layout, Tiingo data layer porting, indicator computation, indicator caching
**Confidence:** HIGH — all findings are from direct inspection of the source files; no external lookups required

---

## Summary

Phase 1 ports three tested, working modules from `rsi_tester/strategy_engine/src/` into a new
`signal_pipeline/src/` package and adds one new module (`cache.py`). All source functions have
been read in full. There are no circular import risks in the planned layout. The config_loader
pattern from the source needs simplification: the new data layer should load API keys directly
from a `.env` file at the project root using `python-dotenv`, without YAML config file
involvement.

The single most important structural decision is the `indicators/` subdirectory. The task
description specifies `src/indicators/indicators.py` (a sub-package), while PLANNING.md §
"New Architecture" shows `src/indicators.py` (a flat file at the `src/` level). These two
layouts are not equivalent and require an explicit planner decision before task creation.
This research recommends the flat layout from PLANNING.md because it matches what all later
phases reference and avoids needless import path complexity.

A second structural note: the `safe_print` function in `composer_signal_generator/main.py`
is a one-liner (`print(*args, **kwargs, flush=True)`). Porting it as a dedicated `utils/`
module is overkill; the planner should decide whether to inline it or omit it entirely.

All runtime dependencies (pandas 2.2.3, numpy 2.2.2, requests, python-dotenv, pyyaml) are
already installed in the project Python environment (Python 3.11.8).

**Primary recommendation:** Use the flat PLANNING.md layout (`src/indicators.py`, not
`src/indicators/indicators.py`). Implement `build_indicator_cache()` operating on a
pre-loaded per-ticker `pd.Series` dict, not on CSV files directly. Keep the `utils/`
module as a stub `__init__.py` only — no `logging.py` needed for Phase 1.

---

## Project Constraints (from CLAUDE.md)

- Think before coding — state assumptions explicitly, surface tradeoffs, ask if unclear.
- Simplicity first — minimum code that solves the problem; no speculative features or
  abstractions for single-use code.
- Surgical changes — touch only what is necessary; do not improve adjacent code or clean
  up pre-existing issues.
- Match existing style even when you would do it differently.
- Remove imports/variables/functions that YOUR changes make unused; leave pre-existing dead
  code alone unless asked.
- No features beyond what was asked. No "flexibility" that was not requested.

---

## Structural Discrepancy: `indicators` Layout

The task description and PLANNING.md disagree on where `indicators.py` lives.

| Source | Layout |
|--------|--------|
| Task description (this research request) | `src/indicators/__init__.py` + `src/indicators/indicators.py` |
| PLANNING.md § "New Architecture" | `src/indicators.py` (flat, no subdirectory) |

**PLANNING.md is authoritative.** Every later phase in PLANNING.md (signal matrix, backtest,
etc.) imports from `src/indicators` as a module, not a package. The flat layout is simpler
and consistent with the rest of the `src/` design, where all modules are single files at the
`src/` level.

**Recommendation for planner:** Use PLANNING.md's flat layout. Do NOT create
`src/indicators/` as a directory. `src/indicators.py` is the correct target.

However, the task description also says `src/utils/logging.py`. PLANNING.md does not mention
a `utils/` subpackage at all. The recommendation is to create `src/utils/__init__.py` as an
empty stub (satisfying the "all directories and placeholder files exist" success criterion)
but not create `logging.py`. Phase 1 uses `print()` directly.

---

## Exact Source File Content and Function Signatures

### `config_loader.py` (full content)

Source: `rsi_tester/strategy_engine/src/config_loader.py` [VERIFIED: direct file read]

```python
import os
import yaml
import json
from pathlib import Path
from dotenv import load_dotenv

def load_config(config_filename="config/strategy_config.yaml", config_dict=None):
    """
    Two calling modes:
      1. load_config("config/my_config.yaml") — loads from YAML file
      2. load_config(config_dict={...})       — accepts pre-built dict, skips file I/O
    In both cases, API keys are loaded from the .env file.
    """
    base_dir = Path(__file__).resolve().parent.parent  # src/ -> strategy_engine/
    env_path = base_dir / ".env"
    load_dotenv(dotenv_path=env_path)
    keys_str = os.getenv("TIINGO_API_KEYS")
    if not keys_str:
        raise ValueError("TIINGO_API_KEYS not found in the environment or .env file.")
    try:
        if keys_str.strip().startswith("["):
            api_keys = json.loads(keys_str)
        else:
            api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    except Exception as e:
        raise ValueError(f"Failed to parse TIINGO_API_KEYS: {e}")
    if not api_keys:
        raise ValueError("API keys list is empty.")
    if config_dict is not None:
        return config_dict, api_keys
    config_path = base_dir / config_filename
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config, api_keys
```

**What the new `src/` needs from this:** Only the API key loading portion. The YAML config
file loading is not needed in the pipeline (the pipeline uses a `config.py` module, not YAML).
The new `src/data/loader.py` should contain a standalone `load_api_keys()` function that
replicates the key-loading half of `load_config()` without the YAML machinery. See proposed
signature below.

### `data_loader.py` — Function Signatures [VERIFIED: direct file read]

Source: `rsi_tester/strategy_engine/src/data_loader.py`

```python
def get_latest_tiingo_date(api_keys: list[str]) -> str:
    """
    Fetches the most recent trading date available on Tiingo using SPY.
    Checks the last 10 days. Returns 'YYYY-MM-DD' string.
    Raises Exception if all keys fail.
    """

def download_ticker_data(ticker: str, api_keys: list[str], data_dir: Path) -> bool:
    """
    Downloads full historical daily OHLC data for a ticker from Tiingo.
    Rotates API keys on failure. Saves to data_dir/{ticker}.csv.
    CSV format: date,open,high,low,close (adjusted prices).
    Raises Exception if all keys exhausted.
    """

def check_freshness_and_update(tickers: list[str], api_keys: list[str], data_dir: Path) -> None:
    """
    Checks each ticker CSV against the latest market date.
    Downloads and rebuilds the full history if CSV is missing or outdated.
    data_dir must be a Path object (not str).
    """
```

**Import note:** The source file has a try/except import guard:
```python
try:
    from config_loader import load_config
except ImportError:
    from .config_loader import load_config
```
This guard is only needed because the source file's `__main__` block calls `load_config()`.
When porting to `src/data/loader.py`, this import is eliminated entirely. The new `loader.py`
exposes `load_api_keys()` instead, and the `__main__` test block is dropped.

### `data_alignment.py` — Function Signatures [VERIFIED: direct file read]

Source: `rsi_tester/strategy_engine/src/data_alignment.py` (becomes `src/data/alignment.py`)

```python
def load_ticker_csv(ticker: str, data_dir: Path) -> pd.DataFrame:
    """
    Reads data_dir/{ticker}.csv. Parses 'date' as datetime, sorts chronologically.
    Returns df with columns ['date', 'open', 'high', 'low', 'close'] if OHLC present,
    else ['date', 'close'] for legacy CSVs.
    Raises FileNotFoundError if CSV missing.
    """

def build_master_dataframe(
    signal_ticker: str,
    target_ticker: str,
    benchmark_ticker: str,
    data_dir: Path,
    filter_assets: list[str] | None = None
) -> pd.DataFrame:
    """
    Loads signal, target, benchmark, and optional filter tickers.
    Renames columns by role (signal_close, target_close, benchmark_close, {ticker}_close).
    Inner-joins all on 'date'. Drops any rows with NaN. Returns merged DataFrame.
    """
```

**New function to add (per PLANNING.md § 1.2):**
```python
def load_multi_ticker_aligned(tickers: list[str], data_dir: Path) -> pd.DataFrame:
    """
    Loads each ticker CSV, extracts the 'close' column, renames it to the ticker symbol.
    Aligns all on 'date' via inner join, forward-fills, sorts by date.
    Returns DataFrame with columns = tickers, index = date (datetime).
    """
```
This is the function that feeds `build_indicator_cache()` and ultimately the signal matrix.
The existing `build_master_dataframe` uses role-based naming (signal_close, target_close)
which is specific to the old pipeline architecture and is not what the new indicator cache
needs. The new function returns a clean `tickers → close price` DataFrame.

### `indicators.py` — Function Signatures [VERIFIED: direct file read]

Source: `rsi_tester/strategy_engine/src/indicators.py` (becomes `src/indicators.py`)

```python
def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average. Returns series.rolling(window=period).mean()."""

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average. Returns series.ewm(span=period, adjust=False).mean()."""

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's RSI. Uses ewm(alpha=1/period, adjust=False) for gain/loss smoothing.
    Sets first `period` rows to NaN explicitly.
    Handles avg_loss=0 edge case (replaces inf with 100).
    Returns values in [0, 100] range with NaN for warmup rows.
    """

def calculate_cumret(series: pd.Series, period: int) -> pd.Series:
    """Cumulative Return. Returns series.pct_change(periods=period) * 100."""

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Wilder's ATR. Requires three separate Series (high, low, close).
    Uses ewm(alpha=1/period, adjust=False).
    Note: signature differs from single-series indicators — requires special dispatch.
    """

def calculate_bbands_lower(series: pd.Series, period: int, num_std: float = 2.0) -> pd.Series:
    """Lower Bollinger Band: SMA(period) - num_std * rolling_std(period)."""

def add_indicator(df: pd.DataFrame, asset_role: str, indicator_name: str, period: int) -> pd.DataFrame:
    """
    Appends a computed indicator column to the master DataFrame.
    Column naming: {asset_role}_{indicator_name}_{period}.
    ATR requires {asset_role}_high and {asset_role}_low columns to exist.
    Raises ValueError for unsupported indicator names.
    Supported: RSI, SMA, EMA, CUMRET, ATR, BBAND_LOWER.
    """
```

**Import hazard in source:** The source `indicators.py` imports `build_master_dataframe` from
`data_alignment` at module level (inside a try/except). When porting to `src/indicators.py`,
this import must be removed entirely — it is only used by the `__main__` test block. The
ported file should have no imports from `src/data/`.

**New functions to add (per PLANNING.md § 1.3):**
```python
def calculate_maxdd(series: pd.Series, period: int) -> pd.Series:
    """Rolling max drawdown over `period` days."""

def calculate_mareturn(series: pd.Series, period: int) -> pd.Series:
    """Rolling mean of daily returns over `period` days."""
```
These are referenced in PLANNING.md § 1.3 as "from fuzz_tester.py". Their implementations
are not in the `rsi_tester` source files read for this phase — they will need to be written
from scratch or located in the `crescendo/` directory. The planner should flag this as a
task that requires locating the reference implementation or writing it.

---

## Question 1: API Key Loading — How Should It Work?

### Current mechanism (source)

`config_loader.py` anchors to `Path(__file__).parent.parent` (i.e., `strategy_engine/`).
It loads `strategy_engine/.env`. The `.env` file contains `TIINGO_API_KEYS` in one of two
formats:
- Comma-separated: `TIINGO_API_KEYS=key1,key2`
- JSON array: `TIINGO_API_KEYS=["key1","key2"]`

### What the new `src/data/loader.py` needs

A simpler `load_api_keys()` function that anchors to `signal_pipeline/` (the project root)
and loads `signal_pipeline/.env`. The YAML config machinery is not needed.

**Proposed implementation:**
```python
import os
import json
from pathlib import Path
from dotenv import load_dotenv

def load_api_keys() -> list[str]:
    """
    Loads Tiingo API keys from the project root .env file.
    Supports comma-separated or JSON array formats.
    Raises ValueError if keys are missing or unparseable.
    """
    # Anchor to signal_pipeline/ (two levels up from src/data/)
    root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(dotenv_path=root / ".env")
    keys_str = os.getenv("TIINGO_API_KEYS")
    if not keys_str:
        raise ValueError(
            "TIINGO_API_KEYS not found. Create signal_pipeline/.env with: "
            "TIINGO_API_KEYS=your_key_here"
        )
    if keys_str.strip().startswith("["):
        api_keys = json.loads(keys_str)
    else:
        api_keys = [k.strip() for k in keys_str.split(",") if k.strip()]
    if not api_keys:
        raise ValueError("TIINGO_API_KEYS is set but empty.")
    return api_keys
```

**`.env` location:** `signal_pipeline/.env` (project root). No `.env` file currently exists
anywhere in the repo (confirmed by glob search). The planner should include a task to create
`.env.example` with placeholder content.

**`.env` format (from existing source code pattern):**
```
TIINGO_API_KEYS=your_api_key_here
# or multiple keys:
# TIINGO_API_KEYS=key1,key2,key3
```

---

## Question 2: `check_freshness_and_update` Import Path

The success criterion is:
```
python -c "from src.data.loader import check_freshness_and_update; print('ok')"
```

This must be run from `signal_pipeline/` as the working directory. The import works if:
1. `signal_pipeline/src/__init__.py` exists
2. `signal_pipeline/src/data/__init__.py` exists
3. `signal_pipeline/src/data/loader.py` exists and defines `check_freshness_and_update`
4. The command is run with `signal_pipeline/` as cwd

**Exact signature the import criterion implies:**
```python
def check_freshness_and_update(tickers: list[str], api_keys: list[str], data_dir: Path) -> None:
```
This is the same signature as the source. No changes needed.

---

## Question 3: `build_indicator_cache()` Design

### Proposed signature (from PLANNING.md § 1.4)

```python
def build_indicator_cache(
    price_df: pd.DataFrame,
    required: list[tuple[str, str, int]]
) -> dict[tuple[str, str, int], np.ndarray]:
    """
    Pre-computes indicator series for all (ticker, fn_label, window) requests.
    Deduplicates: each unique key is computed exactly once.

    Args:
        price_df:  DataFrame with columns = ticker symbols, index = date (datetime).
                   This is the output of load_multi_ticker_aligned().
        required:  List of (ticker, fn_label, window) tuples.
                   May contain duplicates — each unique key computed once.

    Returns:
        dict mapping (ticker, fn_label, window) -> np.ndarray of shape (n_days,).
        Values are float64. NaN values are preserved (warmup periods, etc.).

    Supported fn_labels: "RSI", "SMA", "EMA", "CUMRET", "ATR", "BBAND_LOWER",
                         "MAXDD", "MARETURN"
    """
```

### Implementation sketch

```python
import numpy as np
from src.indicators import (
    calculate_rsi, calculate_sma, calculate_ema, calculate_cumret,
    calculate_atr, calculate_bbands_lower, calculate_maxdd, calculate_mareturn
)

def _compute_indicator(series: pd.Series, fn: str, window: int,
                       price_df: pd.DataFrame, ticker: str) -> pd.Series:
    fn_upper = fn.upper()
    if fn_upper == "RSI":
        return calculate_rsi(series, window)
    elif fn_upper == "SMA":
        return calculate_sma(series, window)
    elif fn_upper == "EMA":
        return calculate_ema(series, window)
    elif fn_upper == "CUMRET":
        return calculate_cumret(series, window)
    elif fn_upper == "ATR":
        # ATR needs high, low, close — requires OHLC columns in price_df
        # Convention: price_df must have f"{ticker}_high", f"{ticker}_low" columns
        # OR the caller passes a DataFrame with separate OHLC columns
        # See "ATR gotcha" section below.
        raise NotImplementedError("ATR dispatch requires OHLC price_df — see design note")
    elif fn_upper == "BBAND_LOWER":
        return calculate_bbands_lower(series, window)
    elif fn_upper == "MAXDD":
        return calculate_maxdd(series, window)
    elif fn_upper == "MARETURN":
        return calculate_mareturn(series, window)
    else:
        raise ValueError(f"Unknown indicator: {fn!r}")

def build_indicator_cache(
    price_df: pd.DataFrame,
    required: list[tuple[str, str, int]]
) -> dict:
    cache = {}
    seen = set()
    for ticker, fn, window in required:
        key = (ticker, fn, window)
        if key in seen:
            continue           # deduplicate — skip if already computed
        seen.add(key)
        series = price_df[ticker]
        indicator_series = _compute_indicator(series, fn, window, price_df, ticker)
        cache[key] = indicator_series.to_numpy(dtype=float)
    return cache
```

**Data source:** `price_df` is the output of `load_multi_ticker_aligned()` — a DataFrame
with ticker symbols as columns and datetime index. It does NOT read CSVs directly; the caller
loads data first, then passes the DataFrame in. This keeps `cache.py` free of I/O.

**Deduplication mechanism:** A `seen` set of `(ticker, fn, window)` tuples. If the same key
appears twice in `required`, the second occurrence is skipped. The cache dict is keyed by
the same tuple, so callers can look up results identically regardless of how many times the
key appeared in `required`.

---

## ATR Gotcha: Multi-Series Indicator in a Single-Series Cache

`calculate_atr` takes three separate Series arguments (`high`, `low`, `close`) while all
other indicator functions take a single `series` argument. This creates a dispatch problem
in `build_indicator_cache`.

**Two options:**

**Option A (recommended for Phase 1):** ATR is out of scope for Phase 1's `build_indicator_cache`. The success criteria do not mention ATR. Include ATR in `indicators.py` as ported, but defer ATR dispatch in the cache to Phase 2 when the full signal config is available and the DataFrame structure is defined.

**Option B (full support):** `load_multi_ticker_aligned()` includes OHLC columns (not just close), named `{ticker}_open`, `{ticker}_high`, `{ticker}_low`, `{ticker}_close`. Then the ATR dispatch extracts the right columns from `price_df` using the ticker name. This is more complex and requires agreeing on the OHLC column naming scheme before Phase 2.

**Recommendation:** Option A for Phase 1. The planner should add a note that ATR dispatch
will be resolved in Phase 2 when the signal config structure is finalized.

---

## Question 4: Data Directory Location

**Current source:** `rsi_tester/strategy_engine/data/` (confirmed empty — no CSV files exist)

**Planned new location:** `signal_pipeline/data/` (at the project root)

This is the correct choice. The `signal_pipeline/data/` directory:
- Is co-located with the project root where `main.py` and `config.py` will live
- Is separate from the `rsi_tester` sub-project's data directory
- Follows the same pattern as `composer_signal_generator/datasets/`

The `data_dir` parameter passed to `check_freshness_and_update()`, `download_ticker_data()`,
and `load_ticker_csv()` will be `Path("signal_pipeline/data/")` or more precisely
constructed as `Path(__file__).resolve().parent.parent.parent / "data"` from within
`src/data/loader.py` if a default is needed. However, the cleaner approach is to pass
`data_dir` explicitly from `main.py` — do not hard-code it in the data layer.

---

## Question 5: Import Graph (No Circular Dependencies)

```
src/
  __init__.py          (empty)
  data/
    __init__.py        (empty)
    loader.py          imports: os, json, requests, pandas, pathlib.Path, datetime,
                               dotenv.load_dotenv
                       NO imports from src.*
    alignment.py       imports: pandas, pathlib.Path
                       NO imports from src.*
    cache.py           imports: numpy, pandas
                               src.indicators (calculate_* functions)
                       NO imports from src.data.*
  indicators.py        imports: pandas, numpy
                       NO imports from src.*
  utils/
    __init__.py        (empty)
```

**Import graph (arrows = "imports from"):**

```
cache.py ──► indicators.py
```

That is the only intra-package dependency. All other modules are leaf nodes with no
intra-`src/` imports.

**No circular risks.** The structure is a strict DAG:
- `loader.py` and `alignment.py` are independent leaves
- `indicators.py` is an independent leaf
- `cache.py` depends on `indicators.py` only
- Nothing in `data/` imports from `cache.py` or `indicators.py`

---

## Question 6: `safe_print` — Port or Skip?

`safe_print` in `composer_signal_generator/main.py` (line 1375):

```python
def safe_print(*args, **kwargs):
    """Print function that ensures output is flushed immediately"""
    print(*args, **kwargs, flush=True)
```

This is a one-liner wrapper. **Recommendation: do not port it as a module.** The `utils/`
directory should be created as an empty stub (`__init__.py` only) to satisfy the "all
directories and placeholder files exist" success criterion. If flush behavior matters in
later phases, add `flush=True` to `print()` calls at those sites. A dedicated module for
a one-liner adds indirection with no benefit and contradicts the CLAUDE.md simplicity rule.

---

## Architecture Patterns

### Recommended Final `src/` Layout

This is PLANNING.md's layout, corrected for the `indicators.py` flat position:

```
signal_pipeline/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py       # load_api_keys(), check_freshness_and_update(),
│   │   │                   # download_ticker_data(), get_latest_tiingo_date()
│   │   ├── alignment.py    # load_ticker_csv(), build_master_dataframe(),
│   │   │                   # load_multi_ticker_aligned()  ← NEW
│   │   └── cache.py        # build_indicator_cache()  ← NEW
│   ├── indicators.py       # calculate_rsi/sma/ema/cumret/atr/bbands_lower/
│   │                       # maxdd/mareturn, add_indicator
│   └── utils/
│       └── __init__.py     # empty stub
├── data/                   # Tiingo CSV cache (created at runtime)
├── .env                    # TIINGO_API_KEYS (gitignored)
├── .env.example            # placeholder with instructions
├── config.py               # TBD in later phases
└── main.py                 # TBD in later phases
```

### Data Flow for Phase 1

```
.env (TIINGO_API_KEYS)
    │
    ▼
load_api_keys()                   ← src/data/loader.py
    │
    ▼
check_freshness_and_update()      ← src/data/loader.py
  writes → signal_pipeline/data/{ticker}.csv
    │
    ▼
load_multi_ticker_aligned()       ← src/data/alignment.py
  reads ← signal_pipeline/data/{ticker}.csv
  returns pd.DataFrame (columns=tickers, index=date)
    │
    ▼
build_indicator_cache()           ← src/data/cache.py
  calls → indicators.py functions
  returns dict[(ticker, fn, window)] → np.ndarray
```

---

## Common Pitfalls

### Pitfall 1: `data_dir` as `str` vs `Path`

`download_ticker_data()` and `check_freshness_and_update()` use `data_dir / f"{ticker}.csv"`
— this requires `data_dir` to be a `pathlib.Path` object, not a string. Callers that pass a
string will get a `TypeError`. The planner should ensure all call sites construct `data_dir`
with `Path(...)`.

### Pitfall 2: `calculate_rsi` warmup is `period` rows, not `period - 1`

The source sets `rsi.iloc[:period] = np.nan`. For `period=14`, rows 0–13 (14 rows) are NaN.
The success criterion says "NaN for the first 14 rows" — this matches the implementation
exactly. No change needed, but verify the test does `assert rsi.iloc[:14].isna().all()`, not
`rsi.iloc[:13]`.

### Pitfall 3: Import path requires running from `signal_pipeline/`

The success criterion `python -c "from src.data.loader import ..."` only works if the cwd
is `signal_pipeline/`. If run from `C:\Python Projects\` it will fail. The test runner or
developer must `cd signal_pipeline` first. The planner should document this in the test task.

### Pitfall 4: `indicators.py` has a module-level import that must be removed

The source `rsi_tester/strategy_engine/src/indicators.py` imports `build_master_dataframe`
from `data_alignment` at module level (try/except). When ported to `src/indicators.py`, this
import MUST be removed — it is only used in the `__main__` test block, which is also dropped.
Leaving this import would create a cross-dependency that violates the DAG structure above.

### Pitfall 5: `load_multi_ticker_aligned` needs inner join, not outer join

The function must use an inner join across all tickers on `date`. An outer join with
forward-fill can silently extend data into dates where some tickers did not trade (e.g.,
newer ETFs). The indicator cache's `n_days` must be identical for all tickers, so the index
must be the common overlap.

### Pitfall 6: `calculate_maxdd` and `calculate_mareturn` have no source implementation

These two functions are referenced in PLANNING.md § 1.3 as "from fuzz_tester.py" but they
do not exist in any of the source files read for this phase. The `crescendo/` directory was
not examined. The planner must either:
(a) Search `crescendo/` for the reference implementation before writing the task, or
(b) Write these functions from scratch using standard rolling-window formulas

**Recommended formulas if written from scratch:**
```python
def calculate_maxdd(series: pd.Series, period: int) -> pd.Series:
    rolling_max = series.rolling(window=period).max()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown  # values <= 0; more negative = larger drawdown

def calculate_mareturn(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change().rolling(window=period).mean()
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Tiingo API key rotation on 429/failure | custom retry loop | existing `download_ticker_data` already has key rotation |
| RSI calculation | manual diff/rolling math | `calculate_rsi()` (ported from source — Wilder's smoothing is already correct) |
| CSV freshness check | timestamp comparison code | `check_freshness_and_update()` (ported from source — already handles this) |
| Deduplication in cache | complex dict logic | simple `seen: set` + `if key in seen: continue` (4 lines) |
| `.env` loading | `os.getenv()` with manual file parsing | `python-dotenv`'s `load_dotenv()` (already in requirements) |

---

## Environment Availability

| Dependency | Required By | Available | Version |
|------------|-------------|-----------|---------|
| Python | All | Yes | 3.11.8 |
| pandas | loader.py, alignment.py, indicators.py | Yes | 2.2.3 |
| numpy | cache.py, indicators.py | Yes | 2.2.2 |
| requests | loader.py (Tiingo HTTP) | Yes | confirmed installed |
| python-dotenv | loader.py (.env loading) | Yes | confirmed installed |
| pyyaml | NOT needed in new src/ | Yes | installed but unused |

**No missing dependencies.** All packages required for Phase 1 are already installed.

**pyyaml note:** The source `config_loader.py` imports `yaml` for YAML config loading.
The new `loader.py` does not load YAML, so `pyyaml` is not a dependency of Phase 1.
Do not add it to Phase 1's requirements list.

---

## Validation Architecture

### Test Map for Phase 1 Success Criteria

| Criterion | Behavior | Test Type | Command |
|-----------|----------|-----------|---------|
| SC-1 | All dirs and `__init__.py` placeholders exist | filesystem check | `py -c "import src; import src.data; import src.data.loader; import src.data.alignment; import src.data.cache; import src.indicators; import src.utils"` |
| SC-2 | Import path works | import smoke test | `py -c "from src.data.loader import check_freshness_and_update; print('ok')"` |
| SC-3 | RSI(SPY, 14) values in 0–100, NaN for first 14 rows | unit test against cached CSV | manual or pytest |
| SC-4 | `build_indicator_cache` deduplicates | unit test with counter | call with duplicate keys, verify fn called once |

**SC-3 and SC-4 can be run without a live Tiingo API key** if a test CSV for SPY already
exists in `signal_pipeline/data/`. Since the data directory is currently empty, SC-2
and SC-3 require either:
(a) A live Tiingo API key in `signal_pipeline/.env`, or
(b) A fixture SPY CSV committed to the repo for testing purposes

The planner should decide which approach. Option (b) is cleaner for a bootstrapping phase
since it allows testing without network access.

### SC-4 Implementation Pattern

```python
call_count = 0
original_rsi = calculate_rsi

def counting_rsi(series, period):
    global call_count
    call_count += 1
    return original_rsi(series, period)

required = [("SPY", "RSI", 14), ("SPY", "RSI", 14), ("SPY", "RSI", 14)]
cache = build_indicator_cache(price_df, required)
assert call_count == 1
assert len(cache) == 1
```

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `rsi_tester/strategy_engine/data/` is empty; no existing CSVs to migrate | Data Directory section | Low — if CSVs exist there, the planner should copy or symlink them to `signal_pipeline/data/` |
| A2 | `calculate_maxdd` and `calculate_mareturn` are not in any read source file | Pitfall 6 | Medium — if they exist in `crescendo/`, the planner can port them directly instead of writing from scratch |
| A3 | No `.env` file currently exists in the repo | load_api_keys() design | Low — if one exists under `rsi_tester/strategy_engine/`, the developer needs to copy it to `signal_pipeline/` |

---

## Open Questions

1. **Flat vs. sub-package `indicators` layout**
   - What we know: Task description says `src/indicators/indicators.py`; PLANNING.md says `src/indicators.py` flat
   - What's unclear: Which was intended as authoritative?
   - Recommendation: Use flat layout from PLANNING.md; confirm with planner before creating directory structure

2. **`calculate_maxdd` and `calculate_mareturn` — source location**
   - What we know: Referenced in PLANNING.md § 1.3 as "from fuzz_tester.py"
   - What's unclear: `crescendo/` directory was not examined — these functions may be there
   - Recommendation: Planner should check `crescendo/` before writing the indicators task; if not found, use the standard rolling formulas documented in Pitfall 6

3. **ATR in `build_indicator_cache` — Phase 1 or Phase 2?**
   - What we know: ATR requires 3 separate series; the cache interface takes one series per ticker
   - What's unclear: Whether Phase 1 success criteria require ATR to be dispatchable through `build_indicator_cache`
   - Recommendation: Exclude ATR from cache dispatch in Phase 1; Phase 2 resolves the OHLC column naming scheme

4. **Test fixture CSV for SC-3 and SC-4**
   - What we know: The data directory is empty; SC-3 requires a real SPY CSV
   - What's unclear: Whether to test against a live API call or a committed fixture
   - Recommendation: Commit a small SPY fixture CSV (e.g., 3 years of daily data) to the repo under `tests/fixtures/` so SC-3 can run offline

---

## Sources

### Primary (HIGH confidence)
- Direct read of `rsi_tester/strategy_engine/src/config_loader.py` — full content transcribed
- Direct read of `rsi_tester/strategy_engine/src/data_loader.py` — all function signatures and logic confirmed
- Direct read of `rsi_tester/strategy_engine/src/data_alignment.py` — all function signatures confirmed
- Direct read of `rsi_tester/strategy_engine/src/indicators.py` — all function signatures confirmed
- Direct read of `PLANNING.md` — architecture layout, phase specs, data flow confirmed
- Direct read of `composer_signal_generator/main.py` (lines 1375–1388) — `safe_print` and `safe_input` confirmed
- Python environment probe (`py -c "import pandas..."`) — package versions confirmed

### Secondary (MEDIUM confidence)
- PLANNING.md § "New Architecture" project structure — authoritative for layout decisions but written before detailed task breakdown (hence the discrepancy with task description)

---

## Metadata

**Confidence breakdown:**
- Source function signatures: HIGH — read directly from source files
- Import graph / no circular deps: HIGH — traced manually from all imports
- `build_indicator_cache` design: HIGH — follows PLANNING.md § 1.4 spec exactly
- `load_api_keys` design: HIGH — direct simplification of `config_loader.py` with path adjusted
- `calculate_maxdd` / `calculate_mareturn` implementations: LOW — not found in source; recommended formulas are [ASSUMED]

**Research date:** 2026-06-10
**Valid until:** Indefinite — all findings are from local source files, not external APIs or docs
