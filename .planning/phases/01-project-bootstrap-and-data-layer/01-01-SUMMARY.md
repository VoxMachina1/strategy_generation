---
phase: 01-project-bootstrap-and-data-layer
plan: 01
status: complete
completed: 2026-06-10
commits:
  - repo: signal_pipeline
    sha: dd934e5
    message: "feat(phase1-groupA): create src/ package structure, port data layer and indicators"
  - repo: signal_pipeline
    sha: 6f7ed9c
    message: "feat(phase1-groupA): add src/data/ modules and tests/ skeleton (gitignore fix)"
  - repo: signal_pipeline
    sha: 15b2d42
    message: "feat(phase1-groupB): add SPY test fixture and run SC-1 through SC-4 verification"
---

# Summary: Phase 1 — Project Bootstrap and Data Layer

## What was done

Created `signal_pipeline/src/` package from scratch, porting the tested data layer from
`rsi_tester/strategy_engine/src/` into independent new modules with no cross-project imports.

### Files created

| File | Contents |
|------|----------|
| `src/__init__.py` | Empty package root |
| `src/data/__init__.py` | Empty package root |
| `src/data/loader.py` | `load_api_keys()` (new), `get_latest_tiingo_date()`, `download_ticker_data()`, `check_freshness_and_update()` — ported from rsi_tester, config_loader dependency removed |
| `src/data/alignment.py` | `load_ticker_csv()` (OHLC-aware), `build_master_dataframe()` — ported; `load_multi_ticker_aligned()` — new, inner join, ticker-symbol columns |
| `src/data/cache.py` | `build_indicator_cache()` with `seen`-set deduplication; ATR raises `NotImplementedError` (Phase 2) |
| `src/indicators.py` | All 6 ported functions + `add_indicator`; `calculate_maxdd()` ported from fuzz_tester.py; `calculate_mareturn()` written from scratch |
| `src/utils/__init__.py` | Empty stub |
| `tests/__init__.py` | Empty |
| `tests/fixtures/SPY.csv` | 90-row synthetic OHLC fixture (numpy seed 42, random walk ~$400) |
| `.env.example` | API key template with instructions |
| `.gitignore` | Added `/data/` (root-anchored to avoid matching `src/data/`) |

### Key architectural decisions

- **Flat layout**: `src/indicators.py` (not a subdirectory) — matches PLANNING.md
- **No rsi_tester imports**: All `src/` files are independent; no `from rsi_tester...` anywhere
- **`load_api_keys()`** anchors to `signal_pipeline/.env` via 3-level parent traversal from `src/data/loader.py`
- **ATR in cache deferred**: `build_indicator_cache` raises `NotImplementedError` for ATR; Phase 2 resolves OHLC column naming
- **`calculate_maxdd`**: Exact port from `rsi_tester/fuzz_tester.py` (rolling peak-to-trough, positive %)
- **`calculate_mareturn`**: Written from scratch (`series.pct_change().rolling(window).mean()`)

### Gotcha fixed

`.gitignore` initially had `data/` (unanchored), which git interpreted as matching `src/data/` too.
Fixed to `/data/` (root-anchored). The `src/data/` modules had to be committed in a second commit.

## Verification

All four success criteria passed:

| SC | Check | Result |
|----|-------|--------|
| SC-1 | All `src.*` packages importable | PASS |
| SC-2 | `from src.data.loader import check_freshness_and_update` | PASS |
| SC-3 | `calculate_rsi(close, 14)` — NaN for rows 0–13, [0,100] for rows 14+ | PASS |
| SC-4 | `build_indicator_cache` with 3 duplicate keys returns 1-entry dict | PASS |
