# Composer Signal Pipeline

Discovers, validates, and exports trading signals to Composer.Trade.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your Tiingo API key
```

Get a free Tiingo key at https://www.tiingo.com/account/api/token

## Entry points

### Full pipeline

```bash
python main.py
```

Runs all 15 stages: data fetch → signal generation → IS backtest → OOS validation → tail analysis → Composer export → output artifacts.

Key flags:

| Flag | Default | Effect |
|------|---------|--------|
| `--top-n N` | 20 | Signals included in symphony |
| `--no-combos` | off | Skip pairwise combo generation |
| `--no-mc` | off | Skip Monte Carlo simulation |
| `--window-type` | walk_forward | OOS window type (walk_forward / expanding / rolling) |
| `--insert-into FILE` | — | Extend an existing Composer symphony (Mode C) |
| `--insert-mode` | leaf | leaf: append; root: wrap existing strategy as gate payload |
| `--dry-run` | off | Print resolved config and exit |

Output is written to `output/{timestamp}/` and includes:
- `all_signals.csv` — full scored signal universe
- `top_n_signals.csv` — top-N signals with all metrics
- `symphony.json` — importable Composer JSON
- `report.html` — self-contained HTML dashboard

### RSI sweep

```bash
python rsi_search.py
```

Fast RSI parameter sweep across configured tickers. Produces a filtered, sorted CSV without running the full validation pipeline. Useful for exploration before committing to a full run.

### Analysis workshop

```bash
python analysis_workshop.py [output_dir]
```

Interactive filter and sort CLI. If `output_dir` is omitted, uses the most recent run in `output/`. Walks through 7 guided quality filters (hit rate, Sharpe, drawdown, tail concentration, etc.) with sane defaults, then saves a `filtered_{timestamp}.csv`.

## Configuration

Edit `config.py` to change signal tickers, target tickers, RSI parameters, and quality thresholds. The full pipeline reads `RSI_SEARCH_CONFIG` from this file and merges pipeline-specific defaults on top.

Additional validation and pipeline parameters can be overridden via JSON file:

```bash
python main.py --config my_config.json
```

## Tests

```bash
python -m pytest tests/
```

165 tests including an end-to-end integration smoke test that runs all pipeline stages on synthetic data with no network calls.
