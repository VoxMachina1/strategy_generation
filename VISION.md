# Signal Pipeline — Vision Document

**Created:** 2026-06-09  
**Projects covered:** `signal_pipeline/` (discovery → validation → assembly) and `strategy_viewer/` (robustness analysis)

---

## What This Is

Two tools, two purposes, one shared foundation:

- **Signal Pipeline** (`signal_pipeline/`): discovers, validates, and packages trading signals as Composer.Trade strategy JSON.
- **Strategy Viewer** (`strategy_viewer/fuzz_tester/`): analyses the robustness of an existing Composer strategy — parameter fragility and tail-event dependence.

They share a data layer (Tiingo), an indicator library, and a common understanding of the Composer JSON format. Long-term, the Strategy Viewer's tail analysis becomes an automated stage inside the pipeline rather than a separate interactive tool.

---

## The Core Job

Find signals that make money *when the benchmark is hurting* — either:

- **Replacement signals** that fire on bad benchmark days and hold a better asset for that window, or
- **Regime signals** that exit to cash during sustained bad market periods.

The `benchmark_median_return < 0` filter is non-negotiable: a signal that fires on average *good* days for the benchmark adds noise, not value. It answers the question "when am I better off somewhere else?" not "when am I in something that is also going up?"

---

## Two Signal Types

### Type 1 — Replacement Signal

A signal that fires on specific benchmark-hurting days and swaps into a different asset for that window.

- **Duration**: short. Median contiguous "on" block < 20 trading days.
- **Logic**: "When QQQ is in trouble (as measured by [indicator]), hold [X] instead."
- **Evaluation**: does holding X on signal-firing days outperform holding the benchmark? Win rate, median return, tail analysis.
- **Assembly**: leaf-node insertion in the strategy JSON tree. The replacement asset is inserted as an `if-child` child; the original benchmark stays in the `else` branch.
- **Key filter**: `benchmark_median_return < 0`.

### Type 2 — Regime / Timing Signal

A portfolio-level gate that determines whether this is a good or bad time to be in the market at all.

- **Duration**: longer. Median contiguous "on" block ≥ 20 trading days.
- **Logic**: "When signal is ON, hold the strategy. When signal is OFF, go to cash (SGOV/BIL)."
- **Evaluation**: does "long when on, cash when off" beat buy-and-hold of the target? Full Sharpe, walk-forward consistency, regime hit rate, crisis hold-out.
- **Assembly**: root-level wrapping. The entire existing strategy becomes the `if-child` child (condition true); a safe asset (SGOV) sits in the `else` branch.
- **Key filters**: `HitRate_Positive_Sharpe`, crisis hold-out Sharpe, stripped Sharpe.

### Auto-Classification

No user input needed. After Stage 1 discovery, the pipeline computes contiguous "on" blocks for each signal and takes the median block length. Below the threshold (configurable, default 20 trading days) → Type 1. At or above → Type 2.

---

## Five-Stage Pipeline (End State)

```
Stage 1 — Discovery
  Input:  tickers, indicator functions, windows, thresholds (config)
  Output: signal matrix — boolean np.ndarray (n_days × n_signals)
          signal metadata — list of specs parallel to columns

Stage 2 — Walk-Forward Validation
  Rolling, expanding, and holdout windows
  Metrics: HitRate_Positive_Sharpe, Sharpe_CoV, anti-home-run filter,
           crisis hold-out Sharpe, stripped Sharpe
  Output:  ranked candidate set with per-signal OOS metrics

Stage 3 — Regime-Level Analysis
  Identifies contiguous "on" blocks per signal
  Computes: regime count, median duration, regime hit rate, per-episode return
  Feeds Type 1 / Type 2 auto-classification

Stage 4 — Tail Analysis (automated)
  Tail concentration, excess kurtosis, stripped win rate (per signal-day return dist.)
  Currently only in Strategy Viewer (interactive). Pipeline automates it for all signals.
  Scores signals by combined (OOS quality + tail health) rank.

Stage 5 — Assembly
  Type 1: leaf-node insertion into target strategy JSON
  Type 2: root-level wrapping of target strategy JSON
  Round-trip verification: parse generated JSON, re-evaluate, compare to signal_matrix
  Output:  ready-to-import Composer symphony JSON + HTML report
```

---

## Three Integration Modes

These describe how an existing Composer strategy JSON interacts with the pipeline. The pipeline is always capable of building new strategies from scratch; the modes describe additional options.

### Mode A — Bootstrap Discovery *(future)*

Load an existing Composer strategy JSON, extract its ticker universe (all `asset` ticker values and `lhs-val` / `rhs-val` signal tickers from conditions), and use that set as the discovery pool. The existing strategy seeds *what tickers to search over*. At the end, discovered signals can optionally be assembled back into the original JSON.

Best for: "I have a strategy I like — find me signals using the assets it already trades."

### Mode B — Validate Existing Signals *(future)*

Load a Composer strategy JSON, extract all `if-child` conditions as-is (each condition becomes a signal spec: fn, window, ticker, comparator, threshold), and run them through Stages 2–4 without any new discovery. Output: the existing strategy's own conditions scored and ranked by OOS quality and tail health.

Best for: "I want to know which of my live strategy's conditions actually hold up out-of-sample." This is the automated counterpart to what Strategy Viewer does interactively.

### Mode C — Extend Existing Strategy *(implementing first)*

Full discovery run (Stage 1), full validation (Stages 2–4), then assemble winners *into* an existing Composer strategy JSON rather than generating a new one from scratch (Stage 5). The existing strategy structure is preserved; new signals are inserted as Type 1 (leaf-level) or Type 2 (root-level wrapper).

This is the most complete path. It generalises the `rsi_tester` workflow — RSI frontrunner insertion into existing strategies — to all indicator types, both signal types, and full walk-forward validation.

Best for: "I have a working strategy and want to add complementary signals to it."

---

## Crisis Hold-Out Filter

### The Problem

Walk-forward splits may place all historical crisis periods inside training windows. A signal that fires into a crash and fails will still pass `HitRate_Positive_Sharpe` if no OOS window ever contains a crisis. The existing consistency and CoV filters do not catch this.

### Design

A set of fixed crisis epochs is defined in config — hardcoded defaults, fully user-editable:

```python
CRISIS_EPOCHS = [
    ("2008-09-01", "2009-06-30"),   # GFC
    ("2020-02-01", "2020-05-31"),   # COVID crash
    ("2022-01-01", "2022-12-31"),   # Rate hike bear
]
```

Per signal, during Stage 2 evaluation:

| Signal behaviour during epoch | Treatment |
|---|---|
| **Never fires** during the epoch | Neutral. No penalty, no bonus. |
| **Fires** during the epoch | Compute Sharpe on those active days only. Must pass a configurable floor (default 0.0) to proceed. |

This specifically penalises "fires confidently into a crash and loses" — not "is inactive during a crash." A slow-moving regime signal (like a 200-DMA cross) that simply goes quiet at the crash impulse and re-enters on recovery is neutral and unaffected. A signal that lights up right before a sharp drawdown and holds through it is filtered out.

### Stripped Sharpe

Complementary metric: compute aggregate OOS Sharpe with the single best OOS window excluded. Catches signals whose OOS Sharpe is dominated by one extraordinary episode (e.g., the COVID rebound). If stripped Sharpe is still positive, performance is distributed. If it collapses, the signal is a one-hit wonder.

This operates at the window level and complements the existing anti-home-run filter (which operates at the return day level via `Sharpe_p90` cap).

---

## Key Design Decisions (and Why)

### benchmark_median_return < 0 is load-bearing

Every discovered signal must clear this filter. It ensures signals are complementary to the strategy — they fire when the strategy's benchmark is hurting, not when everything is going up together. Without it, the pipeline discovers coincidental correlations, not edges.

### No leverage automation

The user manually swaps unleveraged → leveraged after reviewing signal output (QQQ → TQQQ, SPY → SPXL, etc.). The pipeline discovers and validates against unleveraged tickers; the leverage decision belongs to the trader. This preserves the ability to carefully evaluate the unleveraged signal before committing to the 3× version.

### Non-tradeable signal sources are a future extension

Currently all signals must use tradeable ETF tickers (Tiingo data layer). VIX, yield spreads, breadth indicators, and economic releases are a planned extension. The `SignalSource` interface is designed for this extensibility; tradeable-ETF-only is the current implementation scope.

### Tiingo is the single data source

All tools use Tiingo adjusted close (and OHLC where needed). The data layer is being extended to fetch `adjOpen`, `adjHigh`, `adjLow`, `adjClose` — required for true ATR computation. Old close-only CSVs remain backward-compatible.

### Extended dataset: ETF backfill via index substitution (planned integration)

A future integration will allow tickers that don't have a long enough live history to be extended backwards using substitute index data (e.g., substituting the underlying index's return series for the years before the ETF's inception date). When that integration is active, the pre-inception ("extended") portion of any ticker's price series is **treated as holdout data** on equal footing with the post-`holdout_cutoff` period.

Rationale: the extended period is synthetic — it was not tradeable and may exhibit survivorship or reconstruction bias. Signals should be discovered on the live-history training window only. The extended period then serves as an independent robustness check, just as the holdout period does.

Implementation note for when this integration lands:
- The data layer should tag each date in the price series with `origin: "live" | "extended"`.
- The holdout mask should be `origin == "extended" OR date > holdout_cutoff`.
- All three passes of the pipeline must exclude extended-period dates from training regardless of how far back the series goes.

### MOC execution timing is verified correct

Composer evaluates live price at 3:50PM EST and trades at the 4PM close. Signal at day t → return from close t to close t+1. `shift(-1)` on returns is correct. The `EXECUTION_MODE = "MOC"` setting must not be changed.

---

## Strategy Viewer's Long-Term Role

Strategy Viewer (`fuzz_tester.py`) remains a standalone tool for analysing an existing strategy's parameter fragility. Its tail analysis (`compute_tail_metrics`) is the Stage 4 implementation in the pipeline.

Long-term, a "run through pipeline" option could appear in Strategy Viewer: given a strategy JSON, extract all conditions and run them through the full 5-stage validation, producing a combined fragility + OOS quality + tail health score per condition. This would close the loop: discover → validate → analyse → report, all from a single strategy file.

---

## Pre-Requisite Bug Fixes

These bugs exist in the current code and should be resolved before pipeline work begins. They are independent of the architecture.

| Bug | File | Severity | Fix |
|---|---|---|---|
| Latent NameError in frozen combo universe block | `main.py` lines 3084–3141 | HIGH | Remove the block or port the four missing functions |
| Merge conflict cosmetic artifact | `fuzz_tester.py` lines 501–503 | LOW | Delete 3 duplicate comment lines |
| Drawdown unit mismatch (fraction vs percent) | `monte_carlo_sim/...py` | MEDIUM | Multiply simulation `max_drawdowns × 100` before comparison chart |
| `Total_Trades` counts days, not trade entries | `rsi_tester/metrics.py` | MEDIUM | Count 0→1 signal transitions, not active days |
| `_ATR()` is rolling std dev, not true ATR | `main.py` | MEDIUM | Rename to `_rolling_std`; add true `_ATR` wired to OHLC |
| `_BBANDS()` returns upper band only | `main.py` | LOW | Rename to `_BBAND_UPPER`; add `_BBAND_LOWER` |
| `main2.py` expanding window writes to wrong directory | `main2.py` | HIGH | Delete `main2.py` after back-porting 2 items to `main.py` |
