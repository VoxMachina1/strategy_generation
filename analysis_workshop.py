"""
Analysis Workshop — interactive filter and sort CLI for pipeline output.

Usage:
    python analysis_workshop.py [output_dir]

If output_dir is omitted, the most recently modified subdirectory of output/ is used.
Loads top_n_signals.csv, walks through guided quality filters, saves filtered CSV.
"""

import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_GUIDED_FILTERS = [
    ("Consistency_Score",  0.70, ">="),
    ("N_Iterations",       1,    ">="),
    ("Sharpe_p10",         0.3,  ">"),
    ("Sharpe_Stripped",    0.3,  ">"),
    ("MaxDD_p90",          0.35, "<"),
    ("Tail_Concentration", 0.6,  "<"),
    ("WR_Delta",          -0.10, ">"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prompt(msg: str, default) -> str:
    response = input(f"  {msg} [{default}]: ").strip()
    return response if response else str(default)


def _find_latest_output_dir(base: Path) -> Path:
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        print(f"Error: no subdirectories found in {base}")
        sys.exit(1)
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _apply_op(df: pd.DataFrame, col: str, op: str, threshold: float) -> pd.DataFrame:
    if op == ">":  return df[df[col] >  threshold]
    if op == "<":  return df[df[col] <  threshold]
    if op == ">=": return df[df[col] >= threshold]
    if op == "<=": return df[df[col] <= threshold]
    if op == "==": return df[df[col] == threshold]
    print(f"  Unknown operator {op!r}, skipping.")
    return df


# ---------------------------------------------------------------------------
# Filter stages
# ---------------------------------------------------------------------------

def _precalculate(df: pd.DataFrame) -> pd.DataFrame:
    if "Return_p50" in df.columns and "MaxDD_p90" in df.columns:
        df = df.copy()
        denom = np.abs(df["MaxDD_p90"].replace(0, np.nan))
        df["Median_Calmar"] = (df["Return_p50"] / denom).fillna(0)
        print("  Median_Calmar calculated.")
    return df


def _guided_filters(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 50)
    print(" Guided Filtering")
    print("=" * 50)
    print("Press Enter at each prompt to accept the default.\n")

    for col, default, op in _GUIDED_FILTERS:
        if col not in df.columns:
            continue
        print(f"Filter: {col} {op} ? (currently {len(df)} signals)")
        val_str = _prompt(f"Threshold for {col} {op}", default)
        try:
            threshold = float(val_str)
            df = _apply_op(df, col, op, threshold)
            print(f"  → {len(df)} signals remaining")
        except ValueError:
            print(f"  Invalid input, using default {default}")
            df = _apply_op(df, col, op, float(default))
            print(f"  → {len(df)} signals remaining")

    return df


def _custom_filters(df: pd.DataFrame) -> pd.DataFrame:
    add_more = _prompt("\nAdd custom filters? (y/n)", "n").lower()
    if add_more != "y":
        return df

    numeric_cols = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    print("\n--- Custom Filter Menu ---")
    while True:
        print(f"\nSignals remaining: {len(df)}")
        for i, col in enumerate(numeric_cols):
            print(f"  {i + 1:3d}: {col}")
        choice = input("  Column number (or 'done'): ").strip().lower()
        if choice in ("done", "exit", "q", ""):
            break
        try:
            col = numeric_cols[int(choice) - 1]
            op = _prompt(f"  Operator for {col} (>, <, >=, <=, ==)", ">")
            thr = float(input(f"  Threshold for {col} {op}: ").strip())
            df = _apply_op(df, col, op, thr)
            print(f"  → {len(df)} signals remaining")
        except (ValueError, IndexError):
            print("  Invalid input, skipping.")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Resolve output directory
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        base = Path("output")
        if not base.exists():
            print("Error: no output/ directory found. Run main.py first or pass a path.")
            sys.exit(1)
        output_dir = _find_latest_output_dir(base)
        print(f"Using most recent run: {output_dir}")

    csv_path = output_dir / "top_n_signals.csv"
    if not csv_path.exists():
        print(f"Error: top_n_signals.csv not found in {output_dir}")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} signals from {csv_path}")

    # Pre-calculate derived columns
    df = _precalculate(df)

    # Guided filters
    df = _guided_filters(df)

    if df.empty:
        print("\nNo signals remaining after filtering. Exiting.")
        return

    # Custom filters
    df = _custom_filters(df)

    if df.empty:
        print("\nNo signals remaining after filtering. Exiting.")
        return

    # Sort
    print(f"\n--- Sort ({len(df)} signals remaining) ---")
    sort_cols = list(df.columns)
    default_sort = "Sharpe_p50" if "Sharpe_p50" in sort_cols else sort_cols[0]
    for i, col in enumerate(sort_cols):
        print(f"  {i + 1:3d}: {col}")
    sort_choice = _prompt("Sort column", list(sort_cols).index(default_sort) + 1)
    try:
        sort_col = sort_cols[int(sort_choice) - 1]
    except (ValueError, IndexError):
        sort_col = default_sort
        print(f"  Invalid, using {sort_col}")

    asc_choice = _prompt("Ascending? (y/n)", "n").lower()
    ascending = asc_choice == "y"
    df = df.sort_values(by=sort_col, ascending=ascending)
    print(f"  Sorted by {sort_col} {'ascending' if ascending else 'descending'}")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"filtered_{timestamp}.csv"
    df.to_csv(save_path, index=False)
    print(f"\n✓ Saved {len(df)} signals to:\n  {save_path}")


if __name__ == "__main__":
    main()
