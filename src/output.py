"""
Output and report generation for the Composer Signal Pipeline.

Writes all artifacts to output/{run_timestamp}/:
  all_signals.csv      — every base signal with OOS metrics
  all_combos.csv       — every combo with OOS metrics
  top_n_signals.csv    — top-N filtered signals, all metrics
  rsi_search.csv       — RSI search results (optional)
  symphony.json        — copy-paste ready Composer JSON
  report.html          — self-contained sortable HTML dashboard

Public API
----------
write_output()  — write all artifacts for a completed pipeline run
write_csvs()    — write CSV files only
write_symphony_json() — write symphony.json only
write_report_html()   — write report.html only
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 9.1  CSV outputs
# ---------------------------------------------------------------------------

def write_csvs(
    output_dir: Path,
    all_signals_df: pd.DataFrame,
    all_combos_df: pd.DataFrame | None,
    top_n_df: pd.DataFrame,
    rsi_search_df: pd.DataFrame | None = None,
) -> dict:
    """
    Write CSV output files to output_dir.

    Parameters
    ----------
    output_dir     : directory to write into (created if absent)
    all_signals_df : every base signal with OOS metrics
    all_combos_df  : every combo with OOS metrics (None if combos were not run)
    top_n_df       : top-N filtered and sorted signals
    rsi_search_df  : RSI search results (None if RSI search was not run)

    Returns
    -------
    dict mapping artifact name → absolute path written
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    def _write(df: pd.DataFrame, name: str) -> Path:
        p = output_dir / name
        df.to_csv(p, index=False)
        return p

    paths["all_signals"]  = str(_write(all_signals_df, "all_signals.csv"))
    paths["top_n_signals"] = str(_write(top_n_df, "top_n_signals.csv"))

    if all_combos_df is not None and not all_combos_df.empty:
        paths["all_combos"] = str(_write(all_combos_df, "all_combos.csv"))

    if rsi_search_df is not None and not rsi_search_df.empty:
        paths["rsi_search"] = str(_write(rsi_search_df, "rsi_search.csv"))

    return paths


# ---------------------------------------------------------------------------
# 9.3  Composer JSON
# ---------------------------------------------------------------------------

def write_symphony_json(output_dir: Path, symphony_dict: dict) -> str:
    """
    Write symphony.json to output_dir.

    Parameters
    ----------
    output_dir    : destination directory
    symphony_dict : output of src.composer.build_symphony()

    Returns
    -------
    Absolute path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / "symphony.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(symphony_dict, f, indent=2)
    return str(p)


# ---------------------------------------------------------------------------
# 9.4  HTML dashboard
# ---------------------------------------------------------------------------

def _portfolio_equity_curve(
    top_n_df: pd.DataFrame,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_moc: np.ndarray,
    bil_returns: np.ndarray,
    dates: np.ndarray,
) -> tuple:
    """
    Compute equal-weight combined portfolio equity curve and BIL equity curve.

    The portfolio holds each top-N signal's target simultaneously with equal
    weight. When a signal is off, that weight goes to BIL.

    Returns (dates_list, portfolio_cumret, bil_cumret) — all percent-scale.
    """
    if top_n_df.empty or signal_matrix.shape[1] == 0:
        zeros = np.zeros(len(dates))
        return [str(d)[:10] for d in dates], zeros.tolist(), zeros.tolist()

    top_names = top_n_df["signal_name"].unique() if "signal_name" in top_n_df.columns else []
    n_signals = len(top_names)

    if n_signals == 0:
        zeros = np.zeros(len(dates))
        return list(dates), zeros.tolist(), zeros.tolist()

    # Accumulate equal-weight daily P&L across all top-N signals
    combined_daily = np.zeros(len(dates))
    for sig_name in top_names:
        if sig_name not in signal_names:
            continue
        col_idx = signal_names.index(sig_name)
        sig_col = signal_matrix[:, col_idx]
        daily_pnl = np.where(sig_col, target_returns_moc, bil_returns)
        combined_daily += daily_pnl / n_signals

    # BIL daily returns
    bil_daily = bil_returns

    # Convert to cumulative percent-scale equity curves
    port_curve = (np.cumprod(1.0 + combined_daily) - 1.0) * 100
    bil_curve  = (np.cumprod(1.0 + bil_daily) - 1.0) * 100

    date_strs = [str(d)[:10] for d in dates]
    return date_strs, port_curve.tolist(), bil_curve.tolist()


def _tail_badge(tail_concentration: float) -> str:
    """Return an HTML badge coloured by tail risk level."""
    if tail_concentration >= 0.6:
        colour = "#dc3545"   # red — high tail risk
        label  = "HIGH"
    elif tail_concentration >= 0.4:
        colour = "#fd7e14"   # orange — medium
        label  = "MED"
    else:
        colour = "#28a745"   # green — low
        label  = "LOW"
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 6px;'
        f'border-radius:3px;font-size:0.75em;font-weight:bold">{label}</span>'
    )


def write_report_html(
    output_dir: Path,
    run_config: dict,
    top_n_df: pd.DataFrame,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_moc: np.ndarray,
    bil_returns: np.ndarray,
    dates: np.ndarray,
) -> str:
    """
    Write a self-contained HTML dashboard to output_dir/report.html.

    The report includes:
    - Run configuration summary
    - Top-N signals table with all metrics, sortable by any column
    - Tail analysis per signal (Tail Concentration, Kurtosis, Stripped Win Rate)
    - Combined portfolio equity curve vs BIL (SVG line chart)
    - Fragility/tail scoring colour-coded badges per signal

    Parameters
    ----------
    output_dir          : destination directory
    run_config          : dict of run parameters to display in summary
    top_n_df            : top-N signals DataFrame (output of aggregate_oos_results)
    signal_matrix       : (n_days, n_signals) bool
    signal_names        : list[str], parallel to signal_matrix columns
    target_returns_moc  : (n_days,) float — MOC-shifted target returns
    bil_returns         : (n_days,) float — BIL daily returns
    dates               : (n_days,) — date strings or numpy datetime64

    Returns
    -------
    Absolute path of the written file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    date_strs, port_curve, bil_curve = _portfolio_equity_curve(
        top_n_df, signal_matrix, signal_names,
        target_returns_moc, bil_returns, dates,
    )

    # ---- Config summary table -----------------------------------------------
    config_rows = "".join(
        f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
        for k, v in run_config.items()
    )

    # ---- Top-N signals table ------------------------------------------------
    if top_n_df.empty:
        table_html = "<p><em>No signals passed filters.</em></p>"
    else:
        display_cols = [
            c for c in [
                "signal_name", "target",
                "Sharpe_p50", "Sharpe_p10", "Sharpe_Stripped",
                "Sortino_p50", "Calmar_p50",
                "Return_p50", "MaxDD_p90",
                "Consistency_Score", "N_Iterations",
                "Tail_Concentration", "Excess_Kurtosis", "Stripped_Win_Rate",
            ]
            if c in top_n_df.columns
        ]
        if not display_cols:
            display_cols = list(top_n_df.columns)

        header_cells = "".join(
            f'<th onclick="sortTable({i})" style="cursor:pointer;user-select:none">'
            f'{col.replace("_"," ")} &#8597;</th>'
            for i, col in enumerate(display_cols)
        )

        body_rows = []
        for _, row in top_n_df[display_cols].iterrows():
            cells = []
            for col in display_cols:
                val = row[col]
                if col == "Tail_Concentration" and pd.notna(val):
                    cells.append(f"<td>{float(val):.3f} {_tail_badge(float(val))}</td>")
                elif isinstance(val, float):
                    cells.append(f"<td>{val:.4f}</td>")
                else:
                    cells.append(f"<td>{val}</td>")
            body_rows.append("<tr>" + "".join(cells) + "</tr>")

        table_html = f"""
<div style="overflow-x:auto">
<table id="signalsTable" style="border-collapse:collapse;width:100%;font-size:0.85em">
  <thead style="background:#343a40;color:#fff">
    <tr>{header_cells}</tr>
  </thead>
  <tbody>
    {''.join(body_rows)}
  </tbody>
</table>
</div>
"""

    # ---- Equity curve SVG ---------------------------------------------------
    def _svg_curve(dates_list, port, bil):
        if not dates_list:
            return "<p><em>No data for equity curve.</em></p>"

        w, h = 900, 300
        pad_l, pad_r, pad_t, pad_b = 60, 20, 20, 40
        n = len(port)
        if n < 2:
            return "<p><em>Insufficient data for equity curve.</em></p>"

        all_vals = port + bil
        y_min, y_max = min(all_vals), max(all_vals)
        y_range = y_max - y_min or 1.0

        def xp(i):
            return pad_l + (i / (n - 1)) * (w - pad_l - pad_r)

        def yp(v):
            return pad_t + (1.0 - (v - y_min) / y_range) * (h - pad_t - pad_b)

        def polyline(series, colour):
            pts = " ".join(f"{xp(i):.1f},{yp(v):.1f}" for i, v in enumerate(series))
            return f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>'

        # Y-axis ticks
        y_ticks = np.linspace(y_min, y_max, 5)
        y_axis = "".join(
            f'<text x="{pad_l - 5}" y="{yp(v):.1f}" text-anchor="end" '
            f'font-size="10" fill="#666">{v:.1f}%</text>'
            f'<line x1="{pad_l}" y1="{yp(v):.1f}" x2="{w - pad_r}" y2="{yp(v):.1f}" '
            f'stroke="#eee" stroke-width="1"/>'
            for v in y_ticks
        )

        # X-axis ticks (sample ~5 dates)
        step = max(1, n // 5)
        x_axis = "".join(
            f'<text x="{xp(i):.1f}" y="{h - 5}" text-anchor="middle" '
            f'font-size="9" fill="#666">{dates_list[i][:7]}</text>'
            for i in range(0, n, step)
        )

        legend = (
            f'<rect x="{pad_l}" y="{pad_t}" width="12" height="4" fill="#0066cc"/>'
            f'<text x="{pad_l + 16}" y="{pad_t + 7}" font-size="11">Portfolio</text>'
            f'<rect x="{pad_l + 90}" y="{pad_t}" width="12" height="4" fill="#888"/>'
            f'<text x="{pad_l + 106}" y="{pad_t + 7}" font-size="11">BIL</text>'
        )

        return (
            f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px">'
            f'{y_axis}{x_axis}'
            f'{polyline(bil, "#aaa")}'
            f'{polyline(port, "#0066cc")}'
            f'{legend}</svg>'
        )

    equity_svg = _svg_curve(date_strs, port_curve, bil_curve)

    # ---- Run timestamp and title -------------------------------------------
    run_ts = run_config.get("run_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ---- Sort JS -----------------------------------------------------------
    sort_js = """
<script>
var _sortDir = {};
function sortTable(col) {
  var tbl = document.getElementById('signalsTable');
  if (!tbl) return;
  var tbody = tbl.tBodies[0];
  var rows = Array.from(tbody.rows);
  var dir = (_sortDir[col] = !_sortDir[col]);
  rows.sort(function(a, b) {
    var av = a.cells[col].textContent.replace(/[^0-9.-]/g,'');
    var bv = b.cells[col].textContent.replace(/[^0-9.-]/g,'');
    var af = parseFloat(av), bf = parseFloat(bv);
    var cmp = isNaN(af) || isNaN(bf)
      ? av.localeCompare(bv)
      : af - bf;
    return dir ? cmp : -cmp;
  });
  rows.forEach(function(r){ tbody.appendChild(r); });
}
</script>
"""

    # ---- Assemble HTML -------------------------------------------------------
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal Pipeline — Run Report</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0; padding: 16px; background:#f8f9fa; color:#212529; }}
  h1   {{ font-size: 1.4em; margin-bottom: 4px; }}
  h2   {{ font-size: 1.1em; margin-top: 24px; border-bottom: 1px solid #dee2e6; padding-bottom: 4px; }}
  table {{ border-collapse: collapse; font-size: 0.85em; width: 100%; }}
  th, td {{ border: 1px solid #dee2e6; padding: 5px 8px; white-space: nowrap; }}
  thead tr {{ background: #343a40; color: #fff; }}
  tbody tr:nth-child(even) {{ background: #f2f2f2; }}
  tbody tr:hover {{ background: #d0e8ff; }}
  th {{ cursor: pointer; user-select: none; }}
  .card {{ background:#fff; border:1px solid #dee2e6; border-radius:6px; padding:16px; margin-bottom:16px; }}
  .config-table td {{ padding: 3px 8px; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Signal Pipeline — Run Report</h1>
<p style="color:#666;font-size:0.85em">Generated: {run_ts}</p>

<div class="card">
  <h2>Run Configuration</h2>
  <table class="config-table"><tbody>{config_rows}</tbody></table>
</div>

<div class="card">
  <h2>Top-N Signals <span style="color:#666;font-weight:normal;font-size:0.85em">(click column header to sort)</span></h2>
  {table_html}
</div>

<div class="card">
  <h2>Combined Portfolio vs BIL — Equity Curve</h2>
  {equity_svg}
</div>

{sort_js}
</body>
</html>
"""

    p = output_dir / "report.html"
    with open(p, "w", encoding="utf-8") as f:
        f.write(html)
    return str(p)


# ---------------------------------------------------------------------------
# 9.5  Convenience wrapper
# ---------------------------------------------------------------------------

def write_output(
    base_output_dir: Path,
    run_config: dict,
    all_signals_df: pd.DataFrame,
    top_n_df: pd.DataFrame,
    symphony_dict: dict,
    signal_matrix: np.ndarray,
    signal_names: list,
    target_returns_moc: np.ndarray,
    bil_returns: np.ndarray,
    dates: np.ndarray,
    all_combos_df: pd.DataFrame | None = None,
    rsi_search_df: pd.DataFrame | None = None,
    run_timestamp: str | None = None,
) -> dict:
    """
    Write all pipeline output artifacts for a single run.

    Creates output/{run_timestamp}/ and writes all CSVs, symphony.json,
    and report.html. Returns a dict of artifact paths.

    Parameters
    ----------
    base_output_dir     : parent output directory (e.g. Path("output"))
    run_config          : dict displayed in the report header
    all_signals_df      : every base signal with OOS metrics
    top_n_df            : top-N signals (aggregated OOS)
    symphony_dict       : Composer symphony JSON dict
    signal_matrix       : (n_days, n_signals) bool
    signal_names        : list[str], parallel to columns
    target_returns_moc  : (n_days,) float — MOC-shifted target returns
    bil_returns         : (n_days,) float — BIL daily returns
    dates               : (n_days,) array of dates
    all_combos_df       : optional combo results DataFrame
    rsi_search_df       : optional RSI search results DataFrame
    run_timestamp       : optional override; defaults to now as YYYYMMDD_HHMMSS

    Returns
    -------
    dict mapping artifact name → absolute path
    """
    ts = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = {**run_config, "run_timestamp": ts}

    paths = write_csvs(run_dir, all_signals_df, all_combos_df, top_n_df, rsi_search_df)
    paths["symphony_json"] = write_symphony_json(run_dir, symphony_dict)
    paths["report_html"]   = write_report_html(
        run_dir, run_config, top_n_df,
        signal_matrix, signal_names,
        target_returns_moc, bil_returns, dates,
    )

    print(f"[output] Run artifacts written to {run_dir}/")
    for name, p in paths.items():
        print(f"  {name}: {p}")

    return paths
