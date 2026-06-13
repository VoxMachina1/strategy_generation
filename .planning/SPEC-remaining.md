# Spec: Remaining Work — Mode C, Analysis Workshop, main.py

## Overview

Three deliverables closing out the v1.0 milestone:

1. **Mode C** — `insert_into_symphony()` in `src/composer.py`
2. **`analysis_workshop.py`** — interactive filter/sort CLI at the project root
3. **`main.py`** — full pipeline entrypoint at the project root

---

## 1. Mode C — `insert_into_symphony()`

### What it does

Loads an existing Composer symphony JSON and inserts new signals into it without
replacing the existing logic. Two insertion modes:

- **`"leaf"`** (Type 1): Append new if-blocks as additional children inside the
  first `wt-cash-equal` node found in the existing symphony.

- **`"root"`** (Type 2): Wrap the existing symphony's inner content as the
  payload of a new outer gate signal. The new signal fires → existing strategy
  runs; signal off → `safe_asset`.

### Signature

```python
def insert_into_symphony(
    existing_json: dict,
    top_n_specs: list,
    mode: str,              # "leaf" or "root"
    safe_asset: str = "BIL",
) -> dict:
```

- `existing_json` — a parsed Composer symphony dict (same schema as `build_symphony` output)
- `top_n_specs` — list of `{"signal_name": str, "target_ticker": str, "preconditions": list[str] | None}`
  (same element shape as `build_symphony`)
- Returns a **new dict** — does not mutate `existing_json`

### Leaf mode logic

1. Deep-copy `existing_json`.
2. Find the first node where `node["step"] == "wt-cash-equal"` (depth-first traversal).
3. For each spec, build an if-block via `signal_to_if_child` or `combo_to_if_child`
   (same `parse_combo_name` dispatch as `build_symphony`), then `_wrap_with_preconditions`
   if `spec.get("preconditions")` is non-empty.
4. Append the new if-blocks to that node's `"children"` list.
5. Return the modified copy.

### Root mode logic

The existing symphony has structure:
```json
{"step":"root","rebalance":"daily","children":[
  {"step":"wt-cash-equal","children":[...existing if-blocks...]}
]}
```

When wrapping in root mode, the existing `wt-cash-equal` node (and its children) become
the payload delivered when the new gate signal fires. The result structure is:
```json
{"step":"root","rebalance":"daily","children":[
  {"step":"wt-cash-equal","children":[
    {"step":"if","children":[
      {"step":"if-child","is-else-condition?":false,
       <signal condition fields>,
       "children":[
         {"step":"wt-cash-equal","children":[...original if-blocks...]}
       ]},
      {"step":"if-child","is-else-condition?":true,
       "children":[{"step":"asset","ticker":"BIL"}]}
    ]}
  ]}
]}
```

**Implementation steps:**
1. Deep-copy `existing_json`.
2. Extract the existing `wt-cash-equal` node's `"children"` list (the original if-blocks).
3. For each spec in `top_n_specs` (processed in order, outermost first):
   a. Build the signal condition fields (using `signal_to_if_child` / `combo_to_if_child`
      to get the if-block shape, then extract the true-child's condition fields).
   b. Construct a new true-child with those condition fields and
      `"children": [{"step":"wt-cash-equal","children": current_payload}]`.
   c. `current_payload` is now `[new_if_block]` where `new_if_block` wraps the
      previous `current_payload`.
4. Reconstruct the root: `{"step":"root","rebalance":"daily","children":[{"step":"wt-cash-equal","children": current_payload}]}`.
5. Return it.

With multiple specs: spec[0] is the outermost gate (evaluated first), spec[-1] wraps
directly around the original if-blocks.

### Error handling

- `ValueError` if `mode` is not `"leaf"` or `"root"`.
- `ValueError` if `mode == "leaf"` and no `wt-cash-equal` node is found.
- `ValueError` if `existing_json` does not have `"step": "root"` at the top level.

### Tests (add to `tests/test_composer.py`)

- `test_leaf_appends_to_wt_cash_equal`: existing symphony has 2 if-blocks; after
  leaf insert with 1 spec it has 3.
- `test_leaf_preserves_existing_children`: the original 2 if-blocks are present and unchanged.
- `test_root_wraps_wt_cash_equal_contents`: after root insert with 1 spec, the inner
  `wt-cash-equal` node has 1 if-block whose true-child's `"children"` contains
  another `wt-cash-equal` with the original if-blocks.
- `test_root_multiple_specs_nesting`: 2 specs → 2 layers of wrapping; the original
  if-blocks are inside the innermost `wt-cash-equal`.
- `test_insert_does_not_mutate_original`: `existing_json` is unchanged after either mode.
- `test_invalid_mode_raises`: `mode="diagonal"` raises `ValueError`.
- `test_leaf_no_wt_cash_equal_raises`: a malformed symphony without `wt-cash-equal` raises.

---

## 2. `analysis_workshop.py`

### What it does

Interactive CLI that loads `top_n_signals.csv` from a pipeline output directory,
walks the user through guided quality filters (with sane defaults), then saves a
filtered + sorted CSV to the same output directory.

### Location

Project root: `analysis_workshop.py` (alongside `rsi_search.py` and `main.py`).

### CLI interface

```
python analysis_workshop.py [output_dir]
```

- `output_dir` — path to a pipeline run directory (e.g. `output/20260612_143000`).
  If omitted, the script finds the **most recently modified** subdirectory of `output/`
  by `os.path.getmtime`. Exits with a clear error if `output/` doesn't exist or is empty.
- Exits with a clear error if `top_n_signals.csv` is not found in the target directory.

### Pre-calculated columns (added before filtering)

- `Median_Calmar = Return_p50 / abs(MaxDD_p90)` — uses `np.divide` with NaN on
  zero denominator, fills NaN→0. Only computed if both source columns are present.

Note: `MaxDD_p90` is stored as a positive fraction in the pipeline output (max drawdown
magnitude, e.g. 0.15 = 15%). The `abs()` is defensive, not assumed necessary.

### Filter menu (guided, sequential)

Each step:
1. Prints the filter description and current signal count.
2. Prompts with the default in brackets.
3. Applies the filter and prints the new count.
4. Skips silently if the column is absent from the CSV.

| # | Column | Default | Operator |
|---|--------|---------|----------|
| 1 | `Consistency_Score` | 0.70 | >= |
| 2 | `N_Iterations` | 1 | >= |
| 3 | `Sharpe_p10` | 0.3 | > |
| 4 | `Sharpe_Stripped` | 0.3 | > |
| 5 | `MaxDD_p90` | 0.35 | < |
| 6 | `Tail_Concentration` | 0.6 | < |
| 7 | `WR_Delta` | -0.10 | > |

After the guided filters, offer a free-form custom filter loop: pick column by number →
enter operator (`>`, `<`, `>=`, `<=`, `==`) → enter threshold. Type `done` to exit.

### Sorting and saving

1. Prompt for sort column (default: `Sharpe_p50`) and direction (default: descending).
2. Save to `{output_dir}/filtered_{timestamp}.csv` where timestamp is
   `datetime.now().strftime('%Y%m%d_%H%M%S')`. Timestamp makes re-runs non-colliding.
3. Print the full save path and final row count.

### No external dependencies beyond pandas/numpy/stdlib

---

## 3. `main.py`

### Pipeline stages

```
Stage 1:  Load config
Stage 2:  Fetch / refresh price data        → price_df (DatetimeIndex), dates (np.ndarray)
Stage 3:  Build indicator cache             → indicator_cache
Stage 4:  Generate signal matrix            → signal_matrix, signal_names,
                                              signal_metadata, date_index
Stage 5:  Compute target & BIL returns      → target_returns_dict {ticker: moc_returns},
                                              bil_returns (np.ndarray)
Stage 6:  Batch backtest IS (per target)    → is_results_df
Stage 7:  Generate combos                   → combo_names (skipped if --no-combos)
Stage 8:  Batch backtest combos IS          → combo_is_df (skipped if --no-combos)
Stage 9:  Run OOS validation                → oos_raw_df
Stage 10: Aggregate OOS results             → all_signals_df
Stage 11: Compute tail metrics              → all_signals_df (with tail columns appended)
Stage 12: Select top-N                      → top_n_df, top_n_specs
Stage 13: Build symphony                    → symphony_dict
Stage 14: Monte Carlo                       → mc_results (skipped if --no-mc)
Stage 15: Write output                      → output/{timestamp}/
```

**Stage 5 detail:** For each ticker in `target_tickers + [benchmark_ticker]`, compute
`pct_change()` from `price_df[ticker]`, then call `prepare_moc_returns()` to shift
returns for MOC execution. `bil_returns` is the MOC return series for `benchmark_ticker`.

**Stage 6 detail:** `batch_backtest()` operates on one target ticker at a time. Loop
over `target_tickers`, call `batch_backtest(signal_matrix, target_returns_moc[ticker],
bil_returns, date_index)`, tag results with `target=ticker`, concat into `is_results_df`.

**Stage 11 detail:** For each `(signal_name, target)` row in `all_signals_df`, retrieve
the signal column from `signal_matrix` and the target's OOS return series, call
`tail_metrics()`, and merge the returned keys (renamed to Title_Case:
`tail_concentration→Tail_Concentration`, `wr_delta→WR_Delta`, `tail_score→Tail_Score`,
`excess_kurtosis→Excess_Kurtosis`, `base_win_rate→Base_Win_Rate`,
`stripped_win_rate→Stripped_Win_Rate`) back into `all_signals_df`.

**Stage 12 detail:** Sort `all_signals_df` by `Sharpe_p50` descending, take top-N rows.
`top_n_specs` is `[{"signal_name": row.signal_name, "target_ticker": row.target} for row in top_n_df.itertuples()]`.

### CLI interface

```
python main.py [--config CONFIG] [--output OUTPUT] [--workers N]
               [--window-type {walk_forward,expanding,rolling}]
               [--top-n N] [--no-combos] [--no-mc] [--dry-run]
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--config` | `config.py` (imported as `RSI_SEARCH_CONFIG`) | Path to JSON config file; merges over the imported defaults |
| `--output` | `output/` | Base directory for run output |
| `--workers` | `cpu_count - 1` | Process pool size |
| `--window-type` | `walk_forward` | Validation window type; overrides `config["validation"]["window_type"]` |
| `--top-n` | `20` | Overrides `config["top_n"]` |
| `--no-combos` | off | Skip stages 7–8 |
| `--no-mc` | off | Skip stage 14 |
| `--dry-run` | off | Print the resolved config (after merging CLI flags) and exit 0 without running any pipeline stage |

### Config resolution

`main.py` starts from a baseline dict of all defaults, merges `RSI_SEARCH_CONFIG`
on top, then applies any CLI flag overrides. The effective config is what gets
printed under `--dry-run` and logged at stage 1. The existing `config.py` dict
does not need to be modified — pipeline-specific keys (`validation`, `top_n`,
`run_combos`, `run_mc`, `combo_batch_size`, `top_k_for_combos`) live only in
`main.py`'s defaults.

### Progress output

```
[1/15] Loading config...                    done (0.0s)
[2/15] Fetching price data (5 tickers)...   done (3.2s)
...
```

Count reflects actual number of stages run (15, or fewer if `--no-combos`/`--no-mc`).

### Error handling

- Missing `TIINGO_API_KEY` env var: print message pointing to the env var, exit 1.
- Empty signal matrix after stage 4: print warning, exit 1.
- All unexpected exceptions: print traceback, exit 1.

### No unit tests required

Integration-level; tested by running. Individual stage functions are already tested.

---

## Delivery order

1. `insert_into_symphony()` + 7 new tests in `tests/test_composer.py`
2. `analysis_workshop.py`
3. `main.py`

Commit each separately.
