---
phase: 00-pre-requisite-bug-fixes
plan: 01
status: complete
completed: 2026-06-10
commits:
  - repo: signal_pipeline
    sha: ba8a2da
    message: "fix(phase0-groupA): remove dead code, rename _ATR, add _BBAND_LOWER, fix Total_Trades, verify MC drawdown units"
  - repo: signal_pipeline
    sha: 67047a3
    message: "fix(phase0-groupB): back-port precond_mask threading, add max_workers prompt, delete main2.py"
  - repo: rsi_tester
    sha: 1cdc35c
    message: "fix(phase0-groupC): extend Tiingo data_loader to OHLC, forward OHLC in data_alignment, add ATR and BBAND_LOWER to indicators"
---

# Summary: Phase 0 — Pre-Requisite Bug Fixes

## What was done

All 11 confirmed bugs fixed across three commit groups.

### Group A — `composer_signal_generator/main.py`, `metrics.py`, MC script

| Task | Fix |
|------|-----|
| A1 | Deleted ~89-line unreachable dead code block in `build_portfolio_smart_sharpe` (calls to 4 undefined functions after a `return`) |
| A2 | Renamed `_ATR()` → `_rolling_std()`; updated `_ALLOWED_CALLS` (removed ATR/atr, added ROLLING_STD/rolling_std) |
| A3 | Added `_BBAND_LOWER()` to `main.py`; registered BBAND_LOWER/bband_lower in `_ALLOWED_CALLS`; `_BBANDS` preserved |
| A4 | Fixed `Total_Trades` in `metrics.py`: now counts 0→1 signal transitions (`streak_starts`), not active days; `win_rate` denominator stays `n_active_days` |
| A5 | Runtime unit verification: MC drawdown values confirmed **percent-scale** (formula uses percent-scale inputs → percent-scale output) |
| A5b | Added clarifying comment in `plot_drawdown_distributions` — no numeric fix needed |

### Group B — `composer_signal_generator/main.py` (back-ports from main2.py)

| Task | Fix |
|------|-----|
| B1 | Threaded `precond_mask` through `_run_single_backtest` (7-tuple unpack, per-slice apply), `backtest_signals` (signature + tuple), all 3 runner functions (signature + pass-through), and `run_comprehensive_evaluation` (build mask → pass to runners/holdout; removed `filtered_signals` pre-filter). Resume/checkpoint system preserved. |
| B2 | Added `MAX_WORKERS = 5` module-level constant; prompt in `get_enhanced_user_inputs`; `global MAX_WORKERS` set in `enhanced_main`; both `ProcessPoolExecutor` calls now use `MAX_WORKERS` |
| B3 | Deleted `main2.py` |

### Group C — `rsi_tester` repo (committed to its own git)

| Task | Fix |
|------|-----|
| C1 | `data_loader.py`: extended column selection to `adjOpen, adjHigh, adjLow, adjClose` → renamed to `open, high, low, close` |
| C2 | `data_alignment.py`: `load_ticker_csv()` returns OHLC when present, close-only otherwise (backward compat); `build_master_dataframe()` emits `signal_high/low/open`, `target_high/low/open` conditionally |
| C3 | `indicators.py`: added `calculate_atr()` (Wilder ATR via EWM) and `calculate_bbands_lower()`; wired into `add_indicator()` dispatch with clear error message if OHLC columns absent |

## Notes

- `rsi_tester/` is its own nested git repo — Group C committed there, not in the outer `signal_pipeline` repo.
- Existing ticker CSVs (close-only format) must be deleted before re-running `check_freshness_and_update` to get OHLC downloads. The freshness check uses dates only and will not auto-detect stale column format.
- True ATR in `composer_signal_generator/main.py` is deferred to Phase 1 (requires OHLC from yfinance, an architectural change).

## Verification

All 7 final verification checks passed:
1. All 6 modified files compile without syntax errors
2. `main2.py` deleted
3. `_ATR` (word boundary) — zero hits in main.py
4. `filtered_signals` — zero hits in main.py
5. `total_trades = streak_starts` confirmed in metrics.py
6. `adjHigh` present in data_loader.py column selection
7. `calculate_atr` and `BBAND_LOWER` present in indicators.py
