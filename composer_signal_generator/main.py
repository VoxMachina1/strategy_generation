#!/usr/bin/env python3
"""

Combines IAmCaptainNow's signal lab script with methodology for Holdout Split, Walk-Forward, Expanding Window, and Rolling Window tests from Prairie's script to empower further signal discovery

"""
# === Add these imports at the top of your script ===
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import pickle
from pathlib import Path
# ===================================================

IN_COLAB = False
BASE_OUTPUT_DIR = Path(__file__).parent / "datasets"
import yfinance as yf
import pandas as pd
import numpy as np
import itertools
from ta.momentum import RSIIndicator
from tqdm import tqdm
import sys
import os
from datetime import datetime, timedelta
import quantstats as qs
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize as _minimize
from enum import Enum
from typing import List, Dict, Tuple, Optional
warnings.filterwarnings('ignore')

# Import composer-tools for signal conversion
try:
    from composer_tools.converters.short_codes import generate_symphony_code, InvalidConditionError
    COMPOSER_TOOLS_AVAILABLE = True
except ImportError:
    COMPOSER_TOOLS_AVAILABLE = False
    print("Composer-tools not available - install with: pip install composer-tools")

# Combo analysis modules are now built-in (single-file solution)
COMBO_MODULES_AVAILABLE = True
MAX_WORKERS = 5  # configurable via startup prompt; lower values reduce RAM usage

# Extend quantstats periodically
qs.extend_pandas()

# ========= Preconditions Engine =========
import ast

def _as_results_dict(all_results):
    """Return a normalized results dict with expected keys/shape."""
    import pandas as pd
    if not isinstance(all_results, dict):
        all_results = {}
    # normalize simple method frames
    for k in ('walk_forward', 'expanding', 'rolling'):
        v = all_results.get(k, None)
        if not isinstance(v, pd.DataFrame):
            all_results[k] = pd.DataFrame()
    # normalize holdout (dict with dataframes/strings)
    h = all_results.get('holdout', None)
    if not isinstance(h, dict):
        h = {}
    all_results['holdout'] = {
        'merged_results':  h.get('merged_results',  pd.DataFrame()),
        'robust_results':  h.get('robust_results',  pd.DataFrame()),
        'filtered_results':h.get('filtered_results',pd.DataFrame()),
        'train_period':    h.get('train_period',    ''),
        'test_period':     h.get('test_period',     ''),
        'embargo_days':    h.get('embargo_days',    0),
    }
    return all_results

def _get_results_frame(all_results, key):
    """Robustly fetch a DataFrame from results (returns empty DataFrame if missing)."""
    import pandas as pd
    if isinstance(all_results, dict):
        v = all_results.get(key, None)
        if isinstance(v, pd.DataFrame):
            return v
    return pd.DataFrame()

def _PRICE(prices: pd.DataFrame, tkr: str) -> pd.Series:
    s = prices.get(str(tkr), None)
    if s is None:
        # unknown ticker → all-NaN series aligned to index
        return pd.Series(index=prices.index, dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(float)

def _SMA(prices: pd.DataFrame, tkr: str, n: int) -> pd.Series:
    return _PRICE(prices, tkr).rolling(int(n)).mean()

def _EMA(prices: pd.DataFrame, tkr: str, n: int) -> pd.Series:
    return _PRICE(prices, tkr).ewm(span=int(n), adjust=False).mean()

def _RSI(prices: pd.DataFrame, tkr: str, n: int) -> pd.Series:
    s = _PRICE(prices, tkr)
    return pd.Series(RSIIndicator(s, window=int(n)).rsi().values, index=s.index)

def _BBANDS(prices: pd.DataFrame, tkr: str, n: int, std: float = 2.0) -> pd.Series:
    """Bollinger Bands - returns upper band"""
    s = _PRICE(prices, tkr)
    sma = s.rolling(int(n)).mean()
    std_dev = s.rolling(int(n)).std()
    return sma + (std_dev * float(std))

def _BBAND_LOWER(prices: pd.DataFrame, tkr: str, n: int, std: float = 2.0) -> pd.Series:
    """Bollinger Bands — returns lower band (SMA - std * rolling_std)."""
    s = _PRICE(prices, tkr)
    sma = s.rolling(int(n)).mean()
    std_dev = s.rolling(int(n)).std()
    return sma - (std_dev * float(std))

def _rolling_std(prices: pd.DataFrame, tkr: str, n: int) -> pd.Series:
    """Rolling standard deviation of price."""
    s = _PRICE(prices, tkr)
    return s.rolling(int(n)).std()

def _ZSCORE(prices: pd.DataFrame, tkr: str, n: int) -> pd.Series:
    """Z-score of price relative to rolling mean"""
    s = _PRICE(prices, tkr)
    rolling_mean = s.rolling(int(n)).mean()
    rolling_std = s.rolling(int(n)).std()
    return (s - rolling_mean) / rolling_std

_ALLOWED_CALLS = {"PRICE": _PRICE, "SMA": _SMA, "EMA": _EMA, "RSI": _RSI,
                  "BBANDS": _BBANDS, "BBAND_UPPER": _BBANDS, "BBAND_LOWER": _BBAND_LOWER,
                  "ROLLING_STD": _rolling_std, "ZSCORE": _ZSCORE,
                  "price": _PRICE, "sma": _SMA, "ema": _EMA, "rsi": _RSI,
                  "bbands": _BBANDS, "bband_upper": _BBANDS, "bband_lower": _BBAND_LOWER,
                  "rolling_std": _rolling_std, "zscore": _ZSCORE}

# ---- Defaults for Preconditions (two expressions, combined with AND)
PRECONDITION_DEFAULTS = [
    "PRICE('SPY') > SMA('SPY', 200)",
    "RSI('QQQ', 10) < 80"
]
PRECONDITION_COMBINE_DEFAULT = "AND"
_ALLOWED_NODES = (
    ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp,
    ast.Compare, ast.Name, ast.Load, ast.Call, ast.Constant,
    ast.And, ast.Or, ast.Not, ast.Gt, ast.GtE, ast.Lt, ast.LtE, ast.Eq, ast.NotEq,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.Tuple, ast.List
)

def _safe_eval_precond(expr: str, prices: pd.DataFrame) -> pd.Series:
    """
    Evaluate an expression like:
      PRICE('SPY') > SMA('SPY',200) and RSI('QQQ',14) < 30
    Returns a boolean Series aligned to `prices.index`.
    """
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in precondition: {e.text!r} @ col {e.offset}") from e
    if not all(isinstance(n, _ALLOWED_NODES) for n in ast.walk(tree)):
        bad = {type(n).__name__ for n in ast.walk(tree) if not isinstance(n, _ALLOWED_NODES)}
        raise ValueError(f"Unsupported syntax in precondition ({', '.join(sorted(bad))}).")
    # Build a namespace where function Names resolve to callables that close over `prices`
    ns = {name: (lambda *args, f=fn: f(prices, *args)) for name, fn in _ALLOWED_CALLS.items()}
    try:
        out = eval(compile(tree, "<precond>", "eval"), {"__builtins__": {}}, ns)
    except Exception as e:
        raise ValueError(f"Evaluation failed for: {expr!r} → {e}") from e
    if not isinstance(out, pd.Series):
        # Allow expressions that return scalars by broadcasting
        out = pd.Series(bool(out), index=prices.index)
    # Coerce to boolean with NaNs -> False
    return pd.Series(out.astype(bool)).reindex(prices.index).fillna(False)

def build_precondition_series(prices: pd.DataFrame, preconds: list[str], combine: str = "AND") -> pd.Series:
    """
    Combine multiple precondition expressions with AND / OR.
    """
    if not preconds:
        return pd.Series(True, index=prices.index)
    masks = [_safe_eval_precond(expr, prices) for expr in preconds]
    if combine.upper() == "OR":
        m = masks[0].copy()
        for x in masks[1:]:
            m = m | x
        return m
    # Default AND
    m = masks[0].copy()
    for x in masks[1:]:
        m = m & x
    return m
# ========= /Preconditions Engine =========

# ========================= PRICE DOWNLOAD HELPER =========================
# === progress utils (safe tqdm) ===
try:
    from tqdm.auto import tqdm as _tqdm_orig
    def _tqdm(x=None, **kwargs):
        # Force disable all progress bars
        kwargs['disable'] = True
        if x is not None:
            return x  # Just return the iterable without any progress tracking
        else:
            return range(kwargs.get("total", 0))
except Exception:
    def _tqdm(x=None, **kwargs):
        return x if x is not None else range(kwargs.get("total", 0))

# ---- Prompt manager: ask-once helpers ----
class PromptState:
    def __init__(self):
        self.cache = {}

    def ask_bool_once(self, key, question, default=False, input_fn=input):
        if key in self.cache:
            # Already asked; just return the cached answer
            print(f"{question} [{'Y' if self.cache[key] else 'N'}] (kept)")
            return self.cache[key]
        raw = (input_fn(f"{question} [{'Y' if default else 'N'}]: ") or "").strip().lower()
        val = default if raw == "" else raw.startswith("y")
        self.cache[key] = val
        return val

    def set(self, key, value):
        self.cache[key] = value

    def get(self, key, default=None):
        return self.cache.get(key, default)

    def get_smart_sharpe(self):
        """Convenience method for Smart-Sharpe flag."""
        return self.get("smart_sharpe", False)

PM = PromptState()

def _download_prices_adj_with_bar(tickers, start=None, end=None, desc="Downloading"):
    """Download Adj Close per ticker with a progress bar; robust if some tickers fail."""
    import pandas as pd
    frames = []
    for t in _tqdm(tickers, desc=desc, leave=False):
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
                s = df["Adj Close"].rename(t)
            else:
                s = (df.get("Close") or pd.Series(index=pd.DatetimeIndex([]))).rename(t)
            frames.append(s.to_frame())
        except Exception:
            # keep going; empty column will be dropped later
            frames.append(pd.Series(name=t, dtype=float).to_frame())
    out = pd.concat(frames, axis=1)
    out = out.dropna(how="all").sort_index()
    return out

def _download_prices(tickers, start=None, end=None, period=None, desc="Downloading prices"):
    """
    Download prices with preference for adjusted closes (total-return series).
    Falls back to regular closes if adjusted not available.
    Includes progress bar for per-ticker downloads.
    """
    if len(tickers) == 1:
        # Single ticker - use batch download
        if period:
            df = yf.download(tickers, period=period, progress=False)
        else:
            df = yf.download(tickers, start=start, end=end, progress=False)
        # Prefer adjusted if present
        if isinstance(df.columns, pd.MultiIndex):
            if ('Adj Close' in df.columns.get_level_values(0)):
                px = df['Adj Close'].copy()
            else:
                px = df['Close'].copy()
        else:
            # single ticker returns a flat frame/series
            px = df.get('Adj Close', df.get('Close')).copy()
        # Ensure DataFrame with columns for tickers
        if isinstance(px, pd.Series):
            px = px.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        return px
    else:
        # Multiple tickers - use per-ticker download with progress bar
        frames = []
        for t in _tqdm(tickers, desc=desc, leave=False):
            try:
                if period:
                    df = yf.download(t, period=period, progress=False)
                else:
                    df = yf.download(t, start=start, end=end, progress=False)
                if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
                    s = df["Adj Close"].rename(t)
                else:
                    s = (df.get("Adj Close") or df.get("Close") or pd.Series(index=pd.DatetimeIndex([]))).rename(t)
                frames.append(s.to_frame())
            except Exception:
                # keep going; empty column will be dropped later
                frames.append(pd.Series(name=t, dtype=float).to_frame())

        out = pd.concat(frames, axis=1)
        out = out.dropna(how="all").sort_index()
        return out

# === HELPER FUNCTION FOR PARALLEL DOWNLOADS (Must be at top-level) ===
def _download_single_ticker_safe(ticker_symbol: str) -> Optional[pd.DataFrame]:
    """
    Robustly downloads history for a single ticker with retries.
    This is a helper function for parallel execution.
    Returns a DataFrame with the price series, or None on failure.
    """
    import yfinance as yf
    import pandas as pd
    import time as _t

    last_err = None
    for delay in (0, 2, 5):
        if delay:
            _t.sleep(delay)
        try:
            df = yf.download(ticker_symbol, period="max", progress=False, auto_adjust=False, threads=False)
            if not isinstance(df, pd.DataFrame) or df.empty:
                raise RuntimeError("yf.download returned empty dataframe")

            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            price_series = df[col].dropna()
            
            # Ensure we return a DataFrame with the ticker as the column name
            return price_series.to_frame(name=ticker_symbol)

        except Exception as e:
            last_err = e
            # Try fallback on certain errors
            try:
                df = yf.Ticker(ticker_symbol).history(period="max", auto_adjust=False)
                if not isinstance(df, pd.DataFrame) or df.empty:
                    raise RuntimeError("Fallback returned empty dataframe")
                col = "Adj Close" if "Adj Close" in df.columns else "Close"
                price_series = df[col].dropna()
                return price_series.to_frame(name=ticker_symbol)
            except Exception:
                continue # Go to the next retry attempt
    
    if last_err:
        print(f"  ✗ All download attempts failed for {ticker_symbol}: {last_err}")
    
    return None


# === NEW PARALLEL VERSION of download_prices_max_debug ===
def download_prices_max_debug(tickers):
    """
    Robust yfinance downloader using a ThreadPoolExecutor for parallel downloads.
    """
    import pandas as pd

    frames = []
    ticker_list = sorted(list(set([str(x).strip().upper() for x in tickers])))
    print(f"\n=== Parallel Download (period=MAX, {len(ticker_list)} tickers) ===")

    # Use ThreadPoolExecutor for I/O-bound tasks like downloading
    # max_workers can be tuned, but 10-16 is often a good starting point for downloads
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Create a dictionary to map futures to their ticker symbols
        future_to_ticker = {executor.submit(_download_single_ticker_safe, t): t for t in ticker_list}

        # Use tqdm to create a progress bar as futures complete
        for future in tqdm(as_completed(future_to_ticker), total=len(ticker_list), desc="Downloading Tickers"):
            ticker = future_to_ticker[future]
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    frames.append(result_df)
                else:
                    # Create an empty placeholder to signify failure but not halt execution
                    frames.append(pd.DataFrame(columns=[ticker]))
            except Exception as exc:
                print(f"  ✗ Ticker {ticker} generated an exception: {exc}")
                frames.append(pd.DataFrame(columns=[ticker]))

    if not frames:
        raise RuntimeError("No series were downloaded successfully.")

    # Concatenate all downloaded frames, sort by date, and handle potential duplicates
    px = pd.concat(frames, axis=1).sort_index()
    px = px.loc[:,~px.columns.duplicated()] # Drop duplicate columns if any arise
    
    if px.dropna(how="all").empty:
        raise RuntimeError("All downloaded series were empty after concatenation.")
        
    print(f"\n✓ Successfully processed {len(frames)} tickers.")
    return px


def availability_report(px):
    import pandas as pd
    rep = []
    for c in px.columns:
        s = px[c].dropna()
        rep.append({
            "ticker": c,
            "rows": int(len(s)),
            "first": (s.index[0].date() if len(s) else None),
            "last":  (s.index[-1].date() if len(s) else None),
        })
    rpt = pd.DataFrame(rep).set_index("ticker")
    print("\n=== Data Availability Report (Raw) ===")
    for t, r in rpt.iterrows():
        print(f"{t:6s}: rows={r['rows']:5d} first={r['first']} last={r['last']}")
    return rpt


# Quick smoke test function for debugging downloads
def quick_smoke_test():
    """Test price downloads with a small set of tickers before running full pipeline."""
    tickers = ["SPY","XLU","XLP","XLY","XLK","ITB","IYT","IWM","DIA","GLD","TLT","QLD"]
    print("=== Quick Smoke Test ===")
    print(f"Testing downloads for {len(tickers)} tickers...")

    try:
        px = download_prices_max_debug(tickers)
        availability_report(px)
        print("\n✓ Smoke test passed - downloads working correctly")
        return px
    except Exception as e:
        print(f"\n✗ Smoke test failed: {e}")
        print("Check yfinance installation and network connectivity")
        return None


def debug_namespace_pollution():
    """Debug function to identify namespace pollution issues."""
    print("=== Namespace Pollution Debug ===")

    # Check for common problematic names
    problematic_names = ['t', 'df', 's', 'col', 'frames', 'px', 'delay', 'last_err']

    for name in problematic_names:
        if name in globals():
            obj = globals()[name]
            obj_type = type(obj)
            if callable(obj):
                print(f"⚠️  WARNING: '{name}' is callable: {obj_type} = {obj}")
            else:
                print(f"✓ '{name}' is not callable: {obj_type} = {obj}")
        else:
            print(f"✓ '{name}' not in globals")

    # Check for yfinance-related pollution
    try:
        import yfinance as yf
        print(f"\n✓ yfinance imported as: {type(yf)}")
    except Exception as e:
        print(f"✗ yfinance import error: {e}")

    # Check for pandas pollution
    try:
        import pandas as pd
        print(f"✓ pandas imported as: {type(pd)}")
    except Exception as e:
        print(f"✗ pandas import error: {e}")

    print("=== End Namespace Debug ===")


def trim_to_overlap(px, min_overlap=60):
    """Strict overlap first; if too small, soft overlap on cols that meet min_overlap."""
    import numpy as np
    ok = px.notna().all(axis=1)
    strict = px.loc[ok]
    if len(strict) >= min_overlap:
        return strict
    # soft: keep columns with enough data, then drop rows with any NA
    cols_ok = [c for c in px.columns if px[c].notna().sum() >= min_overlap]
    soft = px[cols_ok].dropna()
    if len(soft) >= min_overlap and soft.shape[1] > 0:
        print(f"WARNING: strict overlap < {min_overlap}. Using soft overlap across {len(cols_ok)} tickers.")
        return soft
    return strict  # may be empty; caller will handle


# ========================= SINGLE-FILE COMBO & PORTFOLIO SUPerset =========================
import os as _os, numpy as _np, pandas as _pd
from scipy.stats import spearmanr as _spearmanr
from math import isfinite as _isfinite

# ---- 1) Combo row detection
def _is_combo_row(row):
    if 'Combo_Op' in row and isinstance(row['Combo_Op'], str) and row['Combo_Op']:
        return True
    s = str(row.get('Signal',''))
    return ('+AND+' in s) or ('+OR+' in s) or ('A_AND_NOT_B' in s) or ('B_AND_NOT_A' in s) or ('+' in s)

# ---- 2) Parse a combo Signal name into (members, ops)
def _parse_combo_recipe(signal_name):
    """
    Accepts strings like:
      A+AND+B
      A+OR+B
      A+A_AND_NOT_B+B
      A+B_AND_NOT_A+B
      A+AND+B+OR+C  (multi-leg)
    Returns: members [A,B,(C...)], ops ['AND','OR',...]
    """
    tokens = str(signal_name).split('+')
    # collapse gate tokens containing plus, if any show up (defensive)
    members, ops = [], []
    i = 0
    # Expect alternating member/op/member/op/...
    while i < len(tokens):
        tok = tokens[i]
        if tok in ('AND', 'OR', 'A_AND_NOT_B', 'B_AND_NOT_A'):
            ops.append(tok); i += 1; continue
        # otherwise a member (signal key)
        members.append(tok); i += 1
    # sanity: members should be >=2 and ops = len(members)-1
    if len(members) < 2:  # not actually a combo
        return None, None
    return members, ops

# ---- 3) Combine boolean series with the same semantics as your pipeline
def _combine_op(a, b, op):
    if op == 'AND':           return (a & b)
    if op == 'OR':            return (a | b)
    if op == 'A_AND_NOT_B':   return (a & (~b))
    if op == 'B_AND_NOT_A':   return ((~a) & b)
    raise ValueError(f"Unknown op: {op}")

def _combine_recipe(members, ops, signals):
    cur = signals[members[0]]
    for nxt, op in zip(members[1:], ops):
        cur = _combine_op(cur, signals[nxt], op)
    return cur

# ---- 4) Period helpers
def _parse_period_str(s):  # "YYYY-MM-DD to YYYY-MM-DD"
    s = str(s)
    if 'to' not in s: return None, None
    a, b = [x.strip() for x in s.split('to')]
    return _pd.Timestamp(a), _pd.Timestamp(b)

def _slice_prices(price_df, start, end):
    if start is None or end is None: return price_df
    return price_df.loc[(price_df.index >= start) & (price_df.index <= end)]

# ---- 5) Backtest a boolean signal on a given price slice & ticker (uses your EXECUTION_MODE)
def _bt_signal(sig_series, price_slice, ticker, precond_mask=None):
    daily_ret = price_slice.pct_change()[ticker].fillna(0.0)
    sig = sig_series.reindex(price_slice.index).fillna(False)
    # NEW: apply preconditions (align + NaN→False)
    if precond_mask is not None:
        pc = precond_mask.reindex(price_slice.index).fillna(False)
        sig = sig & pc
    sig, aligned = align_signal_and_returns(sig, daily_ret)
    ret = sig * aligned
    mx  = calculate_quantstats_metrics(ret)
    mx.update({'Time in Market': sig.mean(), 'Signal Returns': ret})
    return mx

# ---- 6) Gather windows (method, iteration, test period)
def _enumerate_windows(all_results):
    windows = []  # list of dict: {method, iter, test_period}
    def _collect(df, method, iter_col):
        if df is None or len(df) == 0: return
        for it in sorted(df[iter_col].dropna().unique()):
            rows = df[df[iter_col]==it]
            # any row has Test_Period string
            tp = rows['Test_Period'].iloc[0] if 'Test_Period' in rows.columns and len(rows)>0 else None
            windows.append({'Method': method, 'Iteration': int(it), 'Test_Period': tp})
    if isinstance(all_results.get('holdout'), dict):
        h = all_results['holdout']
        if len(h.get('filtered_results', _pd.DataFrame()))>0:
            windows.append({'Method':'Holdout', 'Iteration':1, 'Test_Period': h.get('test_period')})
    _collect(all_results.get('walk_forward'), 'Walk-Forward', 'WF_Iteration')
    _collect(all_results.get('expanding'),    'Expanding',     'EW_Iteration')
    _collect(all_results.get('rolling'),      'Rolling',       'Roll_Iteration')
    return windows

# ---- 7) Build frozen combo universe from a seed window
def _seed_combo_frame(all_results):
    # Prefer Holdout filtered_results; else first available WF/EW/Roll iteration
    if isinstance(all_results.get('holdout'), dict):
        df = all_results['holdout'].get('filtered_results')
        if df is not None and len(df)>0:
            cdf = df[df.apply(_is_combo_row, axis=1)].copy()
            if len(cdf)>0:
                return cdf, 'Holdout'
    # fallbacks
    for key, method, itercol in [('walk_forward','Walk-Forward','WF_Iteration'),
                                 ('expanding','Expanding','EW_Iteration'),
                                 ('rolling','Rolling','Roll_Iteration')]:
        df = all_results.get(key)
        if df is None or len(df)==0: continue
        first_it = sorted(df[itercol].dropna().unique())[0]
        cdf = df[(df[itercol]==first_it) & (df.apply(_is_combo_row, axis=1))].copy()
        if len(cdf)>0: return cdf, method
    return _pd.DataFrame(), None

def _freeze_combos(all_results, top_k=30):
    seed_df, seed_method = _seed_combo_frame(all_results)
    if seed_method is None or len(seed_df)==0:
        return []  # no combos to freeze
    # rank by Smart Sharpe then Total Return (both OOS in that seed test)
    seed_df = seed_df.sort_values(['Smart Sharpe','Total Return'], ascending=[False,False])
    picks = seed_df.head(top_k).copy()
    frozen = []  # list of dicts: {'Signal','Ticker','Members','Ops'}
    seen = set()
    for _, row in picks.iterrows():
        sig = str(row['Signal']); tkr = row['Ticker']
        members, ops = _parse_combo_recipe(sig)
        if not members: continue
        key = (sig, tkr)
        if key in seen: continue
        seen.add(key)
        frozen.append({'Signal': sig, 'Ticker': tkr, 'Members': members, 'Ops': ops})
    return frozen

# ---- 8) Evaluate frozen combos across all windows
def _eval_frozen_combos(frozen, windows, signals, full_prices):
    rows = []
    for w in windows:
        start, end = _parse_period_str(w.get('Test_Period')) if w.get('Test_Period') else (None, None)
        px = _slice_prices(full_prices, start, end) if start is not None else full_prices
        if len(px)==0: continue
        for f in frozen:
            sig_name, tkr = f['Signal'], f['Ticker']
            # All members must exist in the global signals dict
            if any(m not in signals for m in f['Members']):
                continue
            combo_series = _combine_recipe(f['Members'], f['Ops'], signals)
            mx = _bt_signal(combo_series, px, tkr)
            rows.append({
                'Method': w['Method'], 'Iteration': w['Iteration'],
                'Test_Period': w.get('Test_Period'),
                'Signal': sig_name, 'Ticker': tkr,
                'Total Return': mx['Total Return'], 'Smart Sharpe': mx['Smart Sharpe'],
                'Sharpe Ratio': mx.get('Sharpe Ratio', _np.nan),
                'Sortino Ratio': mx.get('Sortino Ratio', _np.nan),
                'Calmar Ratio': mx.get('Calmar Ratio', _np.nan),
                'Max Drawdown': mx['Max Drawdown'],
                'Time in Market': mx['Time in Market'],
            })
    out = _pd.DataFrame(rows)
    if len(out)>0:
        out['PeriodKey'] = out['Method'] + '_' + out['Iteration'].astype(str)
    return out

# ---- 9) Dynamic combos (whatever each window produced)
def _gather_dynamic_combo_oos(all_results):
    """
    Returns long panel of combo OOS metrics across WF/EW/Roll (and holdout combos if any).
    Columns: Method, Iteration, Signal, Ticker, Total Return, Smart Sharpe, Max Drawdown
    """
    # Defend against None all_results
    all_results = all_results or {}

    frames = []
    # Holdout (single test period)
    h = all_results.get('holdout', pd.DataFrame())
    if isinstance(h, dict):
        # Prefer merged_results so combos are visible even if they failed final filters
        for key in ['merged_results', 'filtered_results']:
            fr = h.get(key)
            if fr is None or len(fr) == 0:
                continue
            d = fr[fr.apply(_is_combo_row, axis=1)].copy()
            if len(d) > 0:
                d['Method'] = 'Holdout'
                d['Iteration'] = 1
                d['Test_Period'] = h.get('test_period')
                frames.append(d[['Method','Iteration','Test_Period','Signal','Ticker',
                                 'Total Return','Smart Sharpe','Max Drawdown']])
                break  # done once we found data

    # Multi-period methods
    for key, method_name, iter_col in [
        ('walk_forward','Walk-Forward','WF_Iteration'),
        ('expanding','Expanding','EW_Iteration'),
        ('rolling','Rolling','Roll_Iteration'),
    ]:
        df = all_results.get(key)
        if df is None or len(df) == 0:
            continue
        d = df[df.apply(_is_combo_row, axis=1)].copy()
        if len(d) == 0:
            continue
        d['Method'] = method_name
        if iter_col not in d.columns:
            d[iter_col] = 1
        d.rename(columns={iter_col: 'Iteration'}, inplace=True)
        frames.append(d[['Method','Iteration','Test_Period','Signal','Ticker',
                         'Total Return','Smart Sharpe','Max Drawdown']])

    if not frames:
        return _pd.DataFrame(columns=['Method','Iteration','Test_Period','Signal','Ticker',
                                     'Total Return','Smart Sharpe','Max Drawdown'])

    out = _pd.concat(frames, ignore_index=True)

    # Force numeric (robust to mixed dtypes/strings)
    for c in ['Total Return', 'Smart Sharpe', 'Max Drawdown']:
        out[c] = _pd.to_numeric(out[c], errors='coerce')

    # Stable identifiers
    out['PeriodKey'] = out['Method'] + '_' + out['Iteration'].astype(str)
    out['ComboID'] = out['Signal'].astype(str) + ' @ ' + out['Ticker'].astype(str)
    return out

# === FINAL, ROBUST VERSION of _dist_summary ===
def _dist_summary(oos_df):
    """
    Aggregates per combo across windows. This version uses a robust method
    to calculate IQR that avoids the previous DataFrame shape errors.
    """
    if oos_df is None or oos_df.empty:
        return pd.DataFrame()

    for c in ['Total Return', 'Smart Sharpe', 'Max Drawdown']:
        oos_df[c] = pd.to_numeric(oos_df[c], errors='coerce')

    def _p(series, q):
        x = series.dropna().to_numpy(dtype=float)
        return np.nanpercentile(x, q) if len(x) > 0 else np.nan

    grouped = oos_df.groupby(['Signal', 'Ticker'])
    
    agg_data = {
        'N_Iterations': grouped['Smart Sharpe'].count(),
        'Sharpe_p50': grouped['Smart Sharpe'].apply(_p, q=50),
        'Sharpe_p10': grouped['Smart Sharpe'].apply(_p, q=10),
        'Sharpe_p90': grouped['Smart Sharpe'].apply(_p, q=90),
        'Sharpe_IQR': grouped['Smart Sharpe'].apply(lambda x: _p(x, 75) - _p(x, 25)),
        'Return_p50': grouped['Total Return'].apply(_p, q=50),
        'Return_p10': grouped['Total Return'].apply(_p, q=10),
        'Return_p90': grouped['Total Return'].apply(_p, q=90),
        'MaxDD_p90': grouped['Max Drawdown'].apply(_p, q=90),
        'HitRate_Positive_Sharpe': grouped['Smart Sharpe'].apply(lambda x: (x > 0).mean())
    }
    
    agg = pd.DataFrame(agg_data).reset_index()

    agg = agg.sort_values(['Sharpe_p50', 'Return_p50', 'N_Iterations'],
                          ascending=[False, False, False]).reset_index(drop=True)
    return agg

# ---- 11) Greedy low-correlation shortlist (Spearman across periods)
def _lowcorr_shortlist(oos_df, summary_df, metric='Smart Sharpe', corr_threshold=0.30, max_keep=12):
    if len(summary_df)==0:
        return _pd.DataFrame(columns=list(summary_df.columns)+['Selected_Rank'])
    mat = oos_df.pivot_table(index='PeriodKey', columns='Signal', values=metric, aggfunc='mean')
    mat = mat.apply(lambda c: c.fillna(c.median()), axis=0)
    cand = summary_df.copy()
    cand = cand.sort_values(['Sharpe_p50','Return_p50','N_Iterations'], ascending=[False,False,False])
    selected = []
    for _, r in cand.iterrows():
        nm = r['Signal']
        if nm not in mat.columns:
            continue
        ok = True
        for s in selected:
            sname = s['Signal']
            try:
                rho, _ = _spearmanr(_pd.to_numeric(mat[nm], errors='coerce'),
                                    _pd.to_numeric(mat[sname], errors='coerce'),
                                    nan_policy='omit')
            except Exception:
                rho = _np.nan
            if _np.isfinite(rho) and abs(rho) > corr_threshold:
                ok = False; break
        if ok:
            selected.append(r)
            if len(selected) >= max_keep: break
    out = _pd.DataFrame(selected).reset_index(drop=True)
    if len(out)>0:
        out['Selected_Rank'] = _np.arange(1, len(out)+1)
    return out

# ---- 12) (Optional) Portfolio weights: ERC and Smart-Sharpe optimizer
def _erc_weights(oos_df, names, metric='Smart Sharpe'):
    # Simple ERC on per-period metric covariance (use ranks to stabilize)
    mat = oos_df.pivot_table(index='PeriodKey', columns='Signal', values=metric, aggfunc='mean')
    mat = mat[names].rank(axis=0).fillna(mat.median(numeric_only=True))
    cov = _pd.DataFrame(_np.cov(mat.fillna(0.0).T), index=names, columns=names)
    # Solve w s.t. each asset risk contribution equal; use iterative heuristic
    w = _np.ones(len(names))/len(names)
    for _ in range(200):
        port_var = float(w @ cov.values @ w)
        if port_var <= 0: break
        mrc = cov.values @ w  # marginal risk contrib
        rc  = w * mrc         # risk contrib
        target = port_var / len(names)
        grad = rc - target
        w = _np.clip(w - 0.05*grad, 1e-8, None)
        w = w / w.sum()
    return _pd.Series(w, index=names)

def _smart_sharpe_opt(oos_df, names, ret_metric='Total Return', risk_metric='Max Drawdown'):
    # Maximize median(ret) / p90(|drawdown|)
    mat_r = oos_df.pivot_table(index='PeriodKey', columns='Signal', values=ret_metric, aggfunc='mean')[names]
    mat_d = oos_df.pivot_table(index='PeriodKey', columns='Signal', values=risk_metric, aggfunc='mean')[names]
    mat_r = mat_r.fillna(mat_r.median())
    mat_d = mat_d.fillna(mat_d.median())
    # very small grid/coordinate search for robustness
    w = _np.ones(len(names))/len(names)
    def obj(w):
        p = (mat_r.values @ w)
        d = _np.abs(mat_d.values @ w)
        num = _np.nanmedian(p); den = _np.nanpercentile(d, 90)
        return - (num / (den if den!=0 else 1e-6))
    best = (obj(w), w.copy())
    for _ in range(2000):
        i = _np.random.randint(len(names))
        step = (_np.random.rand()-0.5)*0.1
        w2 = w.copy(); w2[i] = max(0.0, w2[i] + step)
        w2 = w2 / w2.sum()
        val = obj(w2)
        if val < best[0]:
            best = (val, w2); w = w2
    return _pd.Series(best[1], index=names)

# ---- 13) Enhanced Portfolio Construction (robust, with fallbacks) ----
def _invvol_weights_robust(rets: _pd.DataFrame, cap: float = None) -> _pd.Series:
    """Robust inverse volatility weights with fallback handling."""
    vol = rets.std().values + 1e-12
    w = 1.0 / vol
    w = w / w.sum()
    if cap is not None:
        # project onto cap-simplex
        w = _np.minimum(w, cap)
        if w.sum() <= 1e-10:
            w = _np.ones_like(w) / len(w)
        else:
            w = w / w.sum()
    # Convert to pandas Series with proper index
    if isinstance(w, _np.ndarray):
        w = _pd.Series(w, index=rets.columns, name="weight")
    return w

def _erc_weights_robust(rets: _pd.DataFrame, cap: float = None) -> _pd.Series:
    """
    Risk parity (Equal Risk Contribution) with robust fallback to invvol.
    """
    try:
        cov = _np.cov(rets.T)
        n = cov.shape[0]

        def risk_contrib(w):
            # RC_i = w_i * (Sigma w)_i
            Sigw = cov @ w
            return w * Sigw

        def obj(w):
            rc = risk_contrib(w)
            return ((rc - rc.mean()) ** 2).sum()

        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        bnds = [(0.0, cap if cap is not None else 1.0) for _ in range(n)]
        x0 = _invvol_weights_robust(rets, cap=None)

        res = _minimize(obj, x0, bounds=bnds, constraints=cons, method="SLSQP", options={"maxiter": 500})
        if not res.success:
            raise RuntimeError(res.message)
        w = _np.clip(res.x, 0, None)
        if w.sum() <= 1e-10:
            return _invvol_weights_robust(rets, cap=cap)
        w /= w.sum()
        # Convert to pandas Series with proper index
        if isinstance(w, _np.ndarray):
            w = _pd.Series(w, index=rets.columns, name="weight")
        return w
    except Exception as e:
        print(f"ERC optimizer failed -> {e}. Falling back to invvol.")
        return _invvol_weights_robust(rets, cap=cap)

def _smart_sharpe_opt_robust(rets: _pd.DataFrame, cv_folds=3, cap=0.35, starts=12, method="erc", rng_seed=123):
    """
    Cross-validated weight search maximizing mean out-of-fold Sharpe.
    Falls back gracefully to base methods if optimization fails.
    """
    rng = _np.random.default_rng(rng_seed)
    T = len(rets)
    if T < cv_folds + 20:
        # too short, just do plain weights
        return (_erc_weights_robust if method == "erc" else _invvol_weights_robust)(rets, cap=cap)

    best_w, best_score = None, -1e9
    for k in range(starts):
        # Random split into folds (contiguous folds)
        fold_bounds = _np.linspace(0, T, cv_folds + 1, dtype=int)
        scores = []
        for f in range(cv_folds):
            tr = rets.iloc[:fold_bounds[f+1]]  # expanding style
            te = rets.iloc[fold_bounds[f]:fold_bounds[f+1]]
            if len(tr) < 30 or len(te) < 10:
                continue
            w = (_erc_weights_robust if method == "erc" else _invvol_weights_robust)(tr, cap=cap)
            port = (te @ w)
            # === ADD THIS FINAL TIMEDELTA FIX HERE ===
            port.reset_index(drop=True, inplace=True)
            # =========================================
            mu, sig = port.mean(), port.std()
            sharpe = (mu / (sig + 1e-12)) * _np.sqrt(252.0)
            scores.append(sharpe)
        if not scores:
            continue
        score = _np.mean(scores)
        if score > best_score:
            # Refit on full data for final weights
            w_final = (_erc_weights_robust if method == "erc" else _invvol_weights_robust)(rets, cap=cap)
            best_w, best_score = w_final, score
    if best_w is None:
        best_w = (_erc_weights_robust if method == "erc" else _invvol_weights_robust)(rets, cap=cap)
    return best_w

# ---- 14) Main exporter: writes both dynamic and frozen artifacts (+ optional weights)
def export_all_combo_artifacts(
    all_results, signals, full_prices, output_dir, name_prefix="",
    frozen_top=30, corr_threshold=0.30, shortlist_size=12,
    write_weights=True
):
    # Defend against None all_results
    all_results = all_results or {}
    for k in ('rolling','walk_forward','expanding','holdout'):
        if k not in all_results or all_results[k] is None:
            if k == 'holdout':
                all_results[k] = {
                    'merged_results': pd.DataFrame(),
                    'robust_results': pd.DataFrame(),
                    'filtered_results': pd.DataFrame(),
                    'train_period': '',
                    'test_period': '',
                    'embargo_days': 0
                }
            else:
                all_results[k] = pd.DataFrame()

    combos_dir = _os.path.join(output_dir, "combos")
    _os.makedirs(combos_dir, exist_ok=True)

    # (A) dynamic (whatever each window produced)
    dyn_oos = _gather_dynamic_combo_oos(all_results)
    dyn_oos.to_csv(_os.path.join(combos_dir, "combo_oos_history_dynamic.csv"), index=False)
    dyn_sum = _dist_summary(dyn_oos)
    dyn_sum.to_csv(_os.path.join(combos_dir, "combo_quant_summary_dynamic.csv"), index=False)
    dyn_short = _lowcorr_shortlist(dyn_oos, dyn_sum, metric='Smart Sharpe',
                                   corr_threshold=corr_threshold, max_keep=shortlist_size)
    dyn_short.to_csv(_os.path.join(combos_dir, "combo_lowcorr_shortlist_dynamic.csv"), index=False)

    # (B) frozen (same identities across all windows)
    windows = _enumerate_windows(all_results)
    frozen = _freeze_combos(all_results, top_k=frozen_top)
    frz_oos = _eval_frozen_combos(frozen, windows, signals, full_prices) if frozen else _pd.DataFrame()
    frz_oos.to_csv(_os.path.join(combos_dir, "combo_oos_history_frozen.csv"), index=False)
    frz_sum = _dist_summary(frz_oos)
    frz_sum.to_csv(_os.path.join(combos_dir, "combo_quant_summary_frozen.csv"), index=False)
    frz_short = _lowcorr_shortlist(frz_oos, frz_sum, metric='Smart Sharpe',
                                   corr_threshold=corr_threshold, max_keep=shortlist_size)
    frz_short.to_csv(_os.path.join(combos_dir, "combo_lowcorr_shortlist_frozen.csv"), index=False)

    # (C) optional: portfolio weights for frozen shortlist
    if write_weights and len(frz_short)>0 and len(frz_oos)>0:
        names = frz_short['Signal'].tolist()

        # Build returns matrix for portfolio construction
        rets_mat = frz_oos.pivot_table(index='PeriodKey', columns='Signal', values='Total Return', aggfunc='mean')
        rets_mat = rets_mat[names].fillna(0.0)  # fill NaNs with 0 (inactive periods)

        if len(rets_mat) > 0 and rets_mat.shape[1] > 0:
            # ERC weights (robust with fallback)
            try:
                erc = _erc_weights_robust(rets_mat, cap=0.35)
                _pd.DataFrame({'Signal': erc.index, 'Weight_ERC': erc.values}) \
                    .to_csv(_os.path.join(combos_dir, "portfolio_weights_erc_frozen.csv"), index=False)
            except Exception as e:
                print(f"ERC weights failed: {e}")

            # Smart-Sharpe optimizer (robust with fallback)
            try:
                ss = _smart_sharpe_opt_robust(rets_mat, cv_folds=3, cap=0.35, starts=8, method="erc")
                _pd.DataFrame({'Signal': ss.index, 'Weight_SmartSharpe': ss.values}) \
                    .to_csv(_os.path.join(combos_dir, "portfolio_weights_smartsharpe_frozen.csv"), index=False)
            except Exception as e:
                print(f"Smart-Sharpe optimization failed: {e}")
        else:
            print("No valid returns data for portfolio construction")

    print("[combos] wrote:")
    for fn in [
        "combo_oos_history_dynamic.csv",
        "combo_quant_summary_dynamic.csv",
        "combo_lowcorr_shortlist_dynamic.csv",
        "combo_oos_history_frozen.csv",
        "combo_quant_summary_frozen.csv",
        "combo_lowcorr_shortlist_frozen.csv",
        "portfolio_weights_erc_frozen.csv",
        "portfolio_weights_smartsharpe_frozen.csv",
        "portfolio_weights_ssopt.csv",
        "portfolio_series_ssopt.csv",
        "portfolio_series_equal_weight.csv",
    ]:
        p = _os.path.join(combos_dir, fn)
        if _os.path.exists(p):
            print("  -", p)
# ====================== /SINGLE-FILE COMBO & PORTFOLIO SUPerset ======================

# ===== Blackout ranges (user-prompt + application) =====
import re

def has_plus(series):
    # Literal contains check that's fast and warning-free
    return series.str.contains('+', regex=False, na=False)

# Your default blackout(s)
BLACKOUT_RANGES_DEFAULT = [("2020-03-20", "2020-12-31")]

def _parse_blackout_chunks(user_text: str):
    """
    Accepts input like:
      - 2020-03-20 to 2020-12-31
      - 2020-03-20,2020-12-31
      - 2020-03-20..2020-12-31
    Multiple ranges separated by ';' (or newlines/commas are fine too).
    Returns list of (start_str, end_str).
    """
    if not user_text:
        return []
    ranges = []
    # split on ';' first, but also support multiple separators gracefully
    for chunk in re.split(r"[;|\n]+", user_text):
        chunk = chunk.strip()
        if not chunk:
            continue
        # capture the first two YYYY-MM-DD in the chunk
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", chunk)
        if len(dates) >= 2:
            ranges.append((dates[0], dates[1]))
    return ranges

def get_blackout_ranges_from_user():
    """
    Prompt the user; return a list of (start,end) strings.
    - Enter to accept default
    - 'none' to disable blackout
    - Or specify one or more ranges like:
        2020-03-20 to 2020-12-31; 2008-09-15 to 2009-03-09
    """
    print("\nOptional: Blackout date ranges (mute periods entirely).")
    print("Enter ranges as 'YYYY-MM-DD to YYYY-MM-DD', separated by ';'")
    print(f"Press Enter to use default: {BLACKOUT_RANGES_DEFAULT}")
    print("Type 'none' to disable blackouts.")
    raw = safe_input("Blackout ranges: ", default="").strip()

    if raw.lower() in ("none", "no", "n", "0"):
        return []

    if raw == "":
        # accept default
        return BLACKOUT_RANGES_DEFAULT.copy()

    parsed = _parse_blackout_chunks(raw)
    if not parsed:
        print("⚠️  Could not parse input; using default.")
        return BLACKOUT_RANGES_DEFAULT.copy()

    # sanity: ensure start <= end
    out = []
    for s, e in parsed:
        sdt, edt = pd.to_datetime(s), pd.to_datetime(e)
        if edt < sdt:
            sdt, edt = edt, sdt
        out.append((sdt.strftime("%Y-%m-%d"), edt.strftime("%Y-%m-%d")))
    return out

def get_preconditions_from_user():
    """
    Prompt the user for precondition expressions.
    Returns (preconditions_list, combine_mode)
    """
    print("\nOptional: Signal Preconditions")
    print("Enter filter expressions like: PRICE('SPY') > SMA('SPY',200)")
    print("Multiple expressions separated by ';' (semicolon)")
    print("Examples:")
    print("  PRICE('SPY') > SMA('SPY',200)")
    print("  PRICE('SPY') > SMA('SPY',200); RSI('QQQ',14) < 30")
    print("  SMA('SPY',50) > SMA('SPY',200) or RSI('IWM',2) < 5")
    print(f"Type 'defaults' to use: {PRECONDITION_DEFAULTS} (combine {PRECONDITION_COMBINE_DEFAULT})")
    print("Type 'none' to disable preconditions.")
    print("Press Enter to skip preconditions entirely.")

    raw = safe_input("Preconditions: ", default="").strip()

    if raw.lower() in ("none", "no", "n", "0"):
        return [], "AND"
    if raw.lower() in ("defaults", "default", "d"):
        # Use defaults when explicitly requested
        return PRECONDITION_DEFAULTS[:], PRECONDITION_COMBINE_DEFAULT
    if raw == "":
        # Skip preconditions entirely
        return [], "AND"

    # Split by semicolon and clean up
    preconds = [p.strip() for p in raw.split(';') if p.strip()]
    if not preconds:
        # No valid preconditions entered
        return [], "AND"

    # Ask how to combine them
    print(f"\nFound {len(preconds)} precondition(s). How to combine them?")
    combine = safe_input("Combine with AND or OR [AND]: ", default="AND").strip().upper()
    if combine not in ("AND", "OR"):
        combine = "AND"
    return preconds, combine

def apply_blackout_ranges(price_df: pd.DataFrame, ranges):
    """
    Set prices to NaN inside each blackout window to break the return chain.
    Signals/indicators inside are NaN; returns inside and immediately after each
    window will become 0.0 once your backtest fills NaNs -> 0.0.
    """
    if not ranges:
        return price_df
    df = price_df.copy()
    for (start, end) in ranges:
        s = pd.to_datetime(start)
        e = pd.to_datetime(end)
        mask = (df.index >= s) & (df.index <= e)
        df.loc[mask] = np.nan
    return df
# ===== /Blackout ranges =====

# ---------- PORTFOLIO SERIES FROM FROZEN SHORTLIST (single-file mode) ----------
import os as _os
import numpy as _np
import pandas as _pd

def _is_combo_name(_s):
    if not isinstance(_s, str):
        return False
    # Treat anything that looks like a combo (AND/OR/gated or has '+') as a combo
    return ('+AND+' in _s) or ('+OR+' in _s) or ('A_AND_NOT_B' in _s) or ('B_AND_NOT_A' in _s) or ('+' in _s)

def _pick_method_df(all_results, method_name):
    key = {
        "Walk-Forward": "walk_forward",
        "Expanding":    "expanding",
        "Rolling":      "rolling",
        "Holdout":      "holdout"
    }[method_name]
    if key == "holdout":
        h = all_results.get("holdout")
        if isinstance(h, dict):
            return h.get("filtered_results")
        return None
    return all_results.get(key)

def _extract_combo_returns_panel(all_results, signals_set, method_name):
    """
    Build a daily panel (index=dates, columns=combo signals) for the chosen method's OOS test windows.
    For Walk-Forward this will be non-overlapping; for others we deduplicate if necessary by first occurrence.
    """
    dfm = _pick_method_df(all_results, method_name)
    if dfm is None or len(dfm) == 0:
        return None

    # Filter to shortlisted combos only and keep rows that actually carry a return series
    df = dfm[(dfm['Signal'].isin(signals_set)) & (dfm['Signal'].apply(_is_combo_name))].copy()
    if 'Signal Returns' not in df.columns or len(df) == 0:
        return None

    # Collect Series into a wide panel by aligning on dates per window, then concatenate
    frames = []
    for _, row in df.iterrows():
        sig = row['Signal']
        series = row['Signal Returns']
        if series is None or not hasattr(series, 'index'):
            continue
        s = _pd.Series(_pd.to_numeric(series, errors='coerce'), name=sig)
        frames.append(s.to_frame())

    if not frames:
        return None

    # Concat by columns, then deduplicate duplicate date rows (keep first)
    panel = _pd.concat(frames, axis=1).sort_index()
    panel = panel[~panel.index.duplicated(keep='first')]
    # Drop all-null rows
    panel = panel.dropna(how='all')
    # Keep only columns in signals_set (already enforced, but re-check)
    keep_cols = [c for c in panel.columns if c in signals_set]
    panel = panel[keep_cols]
    return panel

def _load_shortlist_signals(shortlist_csv_path):
    if _os.path.exists(shortlist_csv_path):
        sh = _pd.read_csv(shortlist_csv_path)
        if 'Signal' in sh.columns:
            # Keep order from shortlist as a soft preference
            return list(sh['Signal'].astype(str).values)
    return []

def _load_ssopt_weights(weights_csv_path):
    """
    Expected format: columns ['Signal','Weight'] (extras allowed); returns a dict.
    Also handles Series CSV format (index,value where value column is 'weight' or unnamed).
    """
    if not _os.path.exists(weights_csv_path):
        return {}
    wdf = _pd.read_csv(weights_csv_path)

    # case 1: expected 2-col table
    if 'Signal' in wdf.columns and 'Weight' in wdf.columns:
        d = {str(r['Signal']): float(r['Weight']) for _, r in wdf.iterrows()}
        # Keep only finite weights
        d = {k: v for k, v in d.items() if _np.isfinite(v)}
        return d

    # case 2: Series CSV like index,value where value column is 'weight' or unnamed
    if 'weight' in wdf.columns and wdf.shape[1] == 2:
        name_col = [c for c in wdf.columns if c != 'weight'][0]
        d = {str(r[name_col]): float(r['weight']) for _, r in wdf.iterrows()}
        # Keep only finite weights
        d = {k: v for k, v in d.items() if _np.isfinite(v)}
        return d

    return {}

def _compute_portfolio_series(panel, weights, renormalize_daily=True):
    """
    panel: DataFrame (dates x signals) of simple daily returns (decimals)
    weights: dict {signal -> weight}; if empty -> equal weight across *available* columns
    renormalize_daily:
        True  -> each day, renormalize weights over signals that have non-null data that day
        False -> fixed weights; missing data contributes 0 (exposure < 1 on those days)
    """
    if panel is None or panel.shape[0] == 0 or panel.shape[1] == 0:
        return None

    panel = panel.copy()
    panel_cols = list(panel.columns)

    # Build base weight vector
    if not weights:
        # Equal weight over all columns
        base_w = _pd.Series(1.0 / len(panel_cols), index=panel_cols)
    else:
        base_w = _pd.Series(0.0, index=panel_cols)
        for k, v in weights.items():
            if k in base_w.index:
                base_w.loc[k] = float(v)
        # Normalize to sum 1 if positive mass
        s = base_w.clip(lower=0).sum()
        if s > 0:
            base_w = base_w.clip(lower=0) / s
        else:
            base_w = _pd.Series(1.0 / len(panel_cols), index=panel_cols)

    # Daily renormalization over the signals that actually have data that day
    if renormalize_daily:
        mask = _pd.notna(panel)
        # effective sum of base weights among available signals per day
        eff = (mask * base_w).sum(axis=1)
        # Avoid divide-by-zero
        eff = eff.replace(0, _np.nan)
        # Weighted sum; where eff is NaN (no signals that day), result is 0
        port = (panel.mul(base_w, axis=1)).sum(axis=1) / eff
        port = port.fillna(0.0)
    else:
        # Fixed weights; missing is just 0 contribution
        port = (panel.mul(base_w, axis=1)).sum(axis=1)
        port = port.fillna(0.0)

    return port

def build_and_write_portfolio_series(
    all_results,
    output_dir,
    shortlist_csv,
    weights_ssopt_csv=None,
    method_preference=("Walk-Forward","Expanding","Rolling","Holdout"),
    renormalize_daily=True
):
    """
    Builds and writes:
      - combos/portfolio_series_equal_weight.csv
      - combos/portfolio_series_ssopt.csv   (if weights file present; else skipped)
    based on the frozen low-correlation shortlist (combo_lowcorr_shortlist.csv).
    Uses the first available method in method_preference for the *daily* OOS panel.
    """
    combos_dir = _os.path.join(output_dir, "combos")
    _os.makedirs(combos_dir, exist_ok=True)

    shortlist = _load_shortlist_signals(shortlist_csv)
    if not shortlist:
        print("[portfolio] no shortlist found; skipping portfolio series.")
        return

    # Pick the best available method for daily series
    panel = None
    used_method = None
    for m in method_preference:
        panel = _extract_combo_returns_panel(all_results, set(shortlist), m)
        if panel is not None and panel.shape[0] > 0 and panel.shape[1] > 0:
            used_method = m
            break

    if panel is None:
        print("[portfolio] no per-window daily returns available for shortlisted combos; skipping series.")
        return

    panel = panel.sort_index()

    # Equal-weight series
    ser_eq = _compute_portfolio_series(panel, weights={}, renormalize_daily=renormalize_daily)
    if ser_eq is not None:
        eq_df = _pd.DataFrame({
            "Return": ser_eq,
            "CumReturn": (1.0 + ser_eq).cumprod() - 1.0
        })
        eq_path = _os.path.join(combos_dir, "portfolio_series_equal_weight.csv")
        eq_df.to_csv(eq_path, index_label="Date")
        print(f"[portfolio] wrote equal-weight series from {used_method} windows -> {eq_path}")

    # Smart-Sharpe optimized weights series (if weights provided)
    if weights_ssopt_csv and _os.path.exists(weights_ssopt_csv):
        wmap = _load_ssopt_weights(weights_ssopt_csv)
        ser_ss = _compute_portfolio_series(panel, weights=wmap, renormalize_daily=renormalize_daily)
        if ser_ss is not None:
            ss_df = _pd.DataFrame({
                "Return": ser_ss,
                "CumReturn": (1.0 + ser_ss).cumprod() - 1.0
            })
            ss_path = _os.path.join(combos_dir, "portfolio_series_ssopt.csv")
            ss_df.to_csv(ss_path, index_label="Date")
            print(f"[portfolio] wrote Smart-Sharpe-weighted series from {used_method} windows -> {ss_path}")
    else:
        print("[portfolio] Smart-Sharpe weights csv not found; only equal-weight series was written.")
# ---------- /PORTFOLIO SERIES (single-file) ----------

# Execution timing configuration
# MOC = Market-on-Close (Composer-like): signal@t * return@t+1 (evaluate at close, execute at close)
# NEXT_BAR = Classic next-bar: signal@t-1 * return@t (decide day before, execute next day)
EXECUTION_MODE = "MOC"  # Set to "NEXT_BAR" for classic behavior
#
# To match Composer's EOD execution: keep EXECUTION_MODE = "MOC" (default)
# To match classic backtesting: change to EXECUTION_MODE = "NEXT_BAR"

def align_signal_and_returns(signal, returns):
    """Align signal and returns based on execution mode"""
    if EXECUTION_MODE == "MOC":
        # Signal at t, returns at t+1 (Composer-like EOD execution)
        return signal, returns.shift(-1).fillna(0.0)
    else:  # NEXT_BAR
        # Signal at t-1, returns at t (classic next-bar execution)
        return signal.shift(1).fillna(False), returns

def _cleanup_tqdm():
    """Clean up any open tqdm progress bars to prevent terminal interference"""
    try:
        import tqdm
        # Close any live bars more gently
        for inst in list(getattr(tqdm, "_instances", [])):
            try:
                if hasattr(inst, 'close'):
                    inst.close()
            except Exception:
                pass
        # Don't clear instances aggressively - just close them
        # tqdm._instances.clear()  # Commented out to prevent output interference
    except Exception:
        pass

# Combo ID functionality is now handled by the built-in superset helpers

# Frozen combo evaluation is now handled by the built-in superset helpers

def safe_print(*args, **kwargs):
    """Print function that ensures output is flushed immediately"""
    print(*args, **kwargs, flush=True)

def safe_input(prompt, default=""):
    """Safe input function that ensures prompts are visible and handles edge cases"""
    # Make sure stdout is flushed so the prompt shows up
    print(prompt, end="", flush=True)
    try:
        s = input()
        return s if s.strip() != "" else default
    except EOFError:
        print(f"\n[warn] no input; using default: {default}", flush=True)
        return default

def _parse_periods(prompt_text, default_list):
    """Parse comma-separated integers from user input"""
    raw = input(f"{prompt_text} (comma-separated, blank for {default_list}): ").strip()
    if not raw:
        return list(default_list)
    try:
        vals = sorted(set(int(x) for x in raw.replace(' ', '').split(',') if x))
        if not vals:
            return list(default_list)
        return vals
    except ValueError:
        print("Invalid entry; using defaults:", default_list)
        return list(default_list)

def normalize_price_panel(df):
    """Normalize price panel to ensure one column per ticker"""
    import pandas as pd
    # If there's a MultiIndex, try to slice to a single price field
    if isinstance(df.columns, pd.MultiIndex):
        lower_last = [str(x).lower() for x in df.columns.get_level_values(-1)]
        if "adj close" in lower_last:
            out = df.xs("Adj Close", axis=1, level=-1, drop_level=True)
        elif "close" in lower_last:
            out = df.xs("Close", axis=1, level=-1, drop_level=True)
        else:
            # Fallback: take the first subcolumn per ticker
            out = df.groupby(level=0, axis=1).first()
        df = out

    # If duplicate column names exist (e.g., two 'PSQ' columns), collapse by first non-null
    if df.columns.duplicated().any():
        df = df.T.groupby(level=0).first().T

    # Ensure numeric dtype, sorted index
    df = df.apply(pd.to_numeric, errors="coerce").sort_index()
    return df

def _series(df, t):
    """Guarantee a 1-D Series when accessing a ticker from price data"""
    s = df[t]
    # If df[t] is a DataFrame (duplicate columns), take first column
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.astype(float)

def unique(seq):
    """Remove duplicates while preserving order"""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _fmt_date(d):
    """Robust date formatting that handles various date types"""
    try:
        return pd.to_datetime(d).strftime('%Y-%m-%d')
    except Exception:
        s = str(d)
        return s[:10] if len(s) >= 10 else s

def report_blackout_status(df, label="Backtest Dataset"):
    """Vectorized blackout reporting - checks entire rows for NaN"""
    # blackout where every ticker is NaN
    blackout_mask = df.isna().all(axis=1)
    # active otherwise
    status = np.where(blackout_mask, "BLACKOUT", "ACTIVE")

    # Optional: summarize continuous blackout ranges
    if blackout_mask.any():
        blocks = blackout_mask.ne(blackout_mask.shift()).cumsum()
        ranges = df.index[blackout_mask].to_series().groupby(blocks[blackout_mask]).agg(['first','last'])
        print(f"\n🔍 {label}: {len(ranges)} blackout block(s)")
        for i, r in ranges.iterrows():
            print(f"  • {r['first'].date()} → {r['last'].date()}")
    else:
        print(f"\n🔍 {label}: no blackout rows detected")
    return blackout_mask, status

def _yf_download_with_retry(tickers, **kwargs):
    """Download with retry logic to handle transient network hiccups"""
    import time
    delays = [0, 2, 5]  # seconds
    last_err = None
    for d in delays:
        if d:
            time.sleep(d)
        try:
            df = yf.download(tickers, **kwargs)
            if df is not None and len(df) > 0:
                return df
        except Exception as e:
            last_err = e
    if last_err:
        print(f"[warn] yfinance retries exhausted: {last_err}")
    return pd.DataFrame()

def _yf_download_with_retry_adj_with_bar(tickers, start=None, end=None, period=None, desc="Downloading"):
    """Download Adj Close per ticker with progress bar and retry logic."""
    import time
    frames = []
    for t in _tqdm(tickers, desc=desc, leave=False):
        delays = [0, 2, 5]  # seconds
        last_err = None
        for d in delays:
            if d:
                time.sleep(d)
            try:
                if period:
                    df = yf.download(t, period=period, progress=False)
                else:
                    df = yf.download(t, start=start, end=end, progress=False)
                if isinstance(df, pd.DataFrame) and "Adj Close" in df.columns:
                    s = df["Adj Close"].rename(t)
                else:
                    s = (df.get("Close") or pd.Series(index=pd.DatetimeIndex([]))).rename(t)
                frames.append(s.to_frame())
                break  # success
            except Exception as e:
                last_err = e
                continue
        else:
            # all retries failed for this ticker
            print(f"[warn] Failed to download {t}: {last_err}")
            frames.append(pd.Series(name=t, dtype=float).to_frame())

    out = pd.concat(frames, axis=1)
    out = out.dropna(how="all").sort_index()
    return out

def _is_combo_row(row) -> bool:
    """Detect if a row is a combo (by presence of Combo_Op or combo-like Signal name)."""
    if 'Combo_Op' in row and isinstance(row['Combo_Op'], str) and row['Combo_Op']:
        return True
    s = str(row.get('Signal', ''))
    # our ops appear in names like +AND+, +OR+, +A_AND_NOT_B+, +B_AND_NOT_A+
    return any(tok in s for tok in ["+AND+", "+OR+", "+A_AND_NOT_B+", "+B_AND_NOT_A+"])

def _percentile(a, q):
    a = pd.Series(a).dropna()
    if len(a) == 0:
        return np.nan
    return np.percentile(a, q)

def _iqr(a):
    a = pd.Series(a).dropna()
    if len(a) == 0:
        return np.nan
    return np.percentile(a, 75) - np.percentile(a, 25)

def _hit_rate_pos(x):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return np.nan
    return (x > 0).mean()

def _build_combo_distribution(df_method: pd.DataFrame, method_label: str) -> pd.DataFrame:
    """
    Build empirical distributions for combos within one method's concatenated results
    (e.g., walk_forward_results across iterations).
    """
    if df_method is None or len(df_method) == 0:
        return pd.DataFrame()

    # Keep only combos
    dfc = df_method[df_method.apply(_is_combo_row, axis=1)].copy()
    if len(dfc) == 0:
        return pd.DataFrame()

    # figure out iteration column to compute coverage if needed
    iter_cols = [c for c in ['WF_Iteration','EW_Iteration','Roll_Iteration'] if c in dfc.columns]
    total_iters = {c: dfc[c].nunique() for c in iter_cols}
    total_iters_all = max(total_iters.values()) if len(total_iters) else None

    g = dfc.groupby(['Signal','Ticker'], as_index=False)

    # Build distribution metrics
    out = g.agg({
        'Smart Sharpe':  [np.median, lambda x: _percentile(x,10), lambda x: _percentile(x,90), _iqr, np.std, _hit_rate_pos],
        'Total Return':  [np.median, lambda x: _percentile(x,10), lambda x: _percentile(x,90), _iqr, np.std, _hit_rate_pos],
        'Max Drawdown':  [np.median, lambda x: _percentile(x,10), lambda x: _percentile(x,90), _iqr, np.std],
        'Time in Market':[np.median, np.mean],
        'Robustness_Score':[np.median, np.mean, np.std]
    })

    # Flatten columns
    out.columns = ['Signal','Ticker',
                   'Sharpe_p50','Sharpe_p10','Sharpe_p90','Sharpe_IQR','Sharpe_std','HitRate_Positive_Sharpe',
                   'Return_p50','Return_p10','Return_p90','Return_IQR','Return_std','HitRate_Positive_Return',
                   'MaxDD_p50','MaxDD_p10','MaxDD_p90','MaxDD_IQR','MaxDD_std',
                   'TiM_p50','TiM_mean',
                   'Robust_p50','Robust_mean','Robust_std']

    # counts
    counts = g.size().rename('N_Iterations')
    out = out.merge(counts, on=['Signal','Ticker'], how='left')

    # add method label and coverage if we can infer total iterations
    out.insert(0, 'Method', method_label)
    if total_iters_all:
        out['Iteration_Coverage'] = out['N_Iterations'] / float(total_iters_all)
    else:
        out['Iteration_Coverage'] = np.nan

    # ranking rule: high Sharpe median, strong p10 return, tolerable tails (less negative MDD is better)
    out = out.sort_values(
        by=['Sharpe_p50','Return_p10','MaxDD_p90'],
        ascending=[False, True, False]  # note: Return_p10 higher is better; MaxDD closer to 0 is better → descending
    ).reset_index(drop=True)

    # Save combo-only distribution file
    if len(out) > 0:
        # Note: paths is not available in this function scope, so we'll save it later in the calling function
        pass

    return out
# ---------- NAVIGATION / SAVING HELPERS ----------

from dataclasses import dataclass, field

@dataclass
class RunPaths:
    root: str
    aggregates: str = field(init=False)
    holdout: str = field(init=False)
    walk_forward: str = field(init=False)
    expanding: str = field(init=False)
    rolling: str = field(init=False)
    # per-method iteration subfolders (created on demand)
    iters: dict = field(default_factory=dict)

    def __post_init__(self):
        self.aggregates   = os.path.join(self.root, "aggregates")
        self.holdout      = os.path.join(self.root, "holdout")
        self.walk_forward = os.path.join(self.root, "walk_forward")
        self.expanding    = os.path.join(self.root, "expanding")
        self.rolling      = os.path.join(self.root, "rolling")
        for p in [self.aggregates, self.holdout, self.walk_forward, self.expanding, self.rolling]:
            os.makedirs(p, exist_ok=True)

    def method_dir(self, method: str) -> str:
        return getattr(self, method)

    def iter_dir(self, method: str) -> str:
        d = os.path.join(self.method_dir(method), "iters")
        os.makedirs(d, exist_ok=True)
        return d

manifest_rows = []  # (path, kind, method, note, nrows, ncols)

def save_df(df, folder, filename, kind="", method="", note="", prefix=""):
    os.makedirs(folder, exist_ok=True)
    # Add prefix if provided for better sorting
    if prefix:
        filename = f"{prefix}_{filename}"
    path = os.path.join(folder, filename)
    # drop heavy columns like returns to keep CSV light
    try:
        df_to_save = df.drop(columns=['Signal Returns'], errors='ignore')
    except Exception:
        df_to_save = df
    df_to_save.to_csv(path, index=False)

    # Also save combo-only version if this contains combos
    if len(df_to_save) > 0 and 'Signal' in df_to_save.columns:
        combo_df = df_to_save[has_plus(df_to_save['Signal'])]
        if len(combo_df) > 0:
            combo_filename = filename.replace('.csv', '_combos_only.csv')
            combo_path = os.path.join(folder, combo_filename)
            _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)
            manifest_rows.append([
                os.path.relpath(combo_path), f"{kind}_combos_only", method, f"{note} (combos only)", len(combo_df), combo_df.shape[1]
            ])

    manifest_rows.append([
        os.path.relpath(path), kind, method, note, len(df_to_save), df_to_save.shape[1]
    ])
    return path

# === NEW, MORE HELPFUL README FUNCTION ===
def write_readme(paths: RunPaths, cfg_summary: str):
    """
    Generates a comprehensive README.md file that includes a recommended
    analysis workflow and the detailed configuration for the run.
    """
    # Define the paths to the most important files for the user
    # os.path.join is used to make the paths work on any operating system
    primary_files = {
        "Executive Summary": os.path.join("aggregates", "evaluation_summary.csv"),
        "Deep Dive Analysis": os.path.join("aggregates", "method_averages.csv"),
        "Combo-Specific Stats": os.path.join("combos", "combo_quant_summary_frozen.csv"),
        "Portfolio Shortlist": os.path.join("combos", "combo_lowcorr_shortlist_frozen.csv")
    }

    # Use Path objects for robust link creation
    root_path = Path(paths.root)
    
    # Build the README content section by section
    readme = []
    readme.append("# Backtest Run Summary\n")
    
    readme.append("## Recommended Analysis Workflow\n")
    readme.append("This run generated a lot of data. Here is the recommended way to analyze it, from the highest level to the most detailed.")
    
    readme.append("\n**1. Start with the Executive Summary:**")
    readme.append(f"- **File:** `{primary_files['Executive Summary']}`")
    readme.append("- **Purpose:** This is your 'Top 50' leaderboard. It shows the signals with the best *average* performance across all test windows. It's a great first look at what worked.")
    
    readme.append("\n**2. Do a Deep Dive for Consistency (Most Important Step):**")
    readme.append(f"- **File:** `{primary_files['Deep Dive Analysis']}`")
    readme.append("- **Purpose:** This is the most important file for finding robust, 'all-weather' signals. It shows the median, average, and standard deviation of performance for EVERY signal. Use this to filter for signals that meet your specific criteria for consistency and risk.")
    
    readme.append("\n**3. Analyze Combo Performance Distributions:**")
    readme.append(f"- **File:** `{primary_files['Combo-Specific Stats']}`")
    readme.append("- **Purpose:** This file provides a detailed statistical breakdown for the top *combo* signals. It shows their full performance distribution (p10, p50, p90). **Focus on signals with a high `Sharpe_p50` (good typical performance) and a high `Sharpe_p10` (good worst-case performance).**")
    
    readme.append("\n**4. Review the Pre-Built Portfolio:**")
    readme.append(f"- **File:** `{primary_files['Portfolio Shortlist']}`")
    readme.append("- **Purpose:** This file represents a ready-to-use portfolio of diversified, low-correlation signals selected from the file above. It's an excellent candidate for a final strategy.")

    readme.append("\n## Run Configuration\n")
    readme.append("The following settings were used to generate these results:")
    readme.append("```")
    readme.append(cfg_summary.strip())
    readme.append("```")
    
    readme.append("\n## Full File Manifest\n")
    readme.append("For a complete list of all generated files and their contents, please see the `manifest.csv` file.")

    # Write the content to the README.md file in the root results directory
    try:
        with open(os.path.join(paths.root, "README.md"), "w") as f:
            f.write("\n".join(readme))
    except Exception as e:
        print(f"Warning: Could not write README.md file. Error: {e}")

def write_manifest(paths: RunPaths):
    import pandas as pd
    m = pd.DataFrame(manifest_rows, columns=["file","kind","method","note","rows","cols"])
    m.to_csv(os.path.join(paths.root, "manifest.csv"), index=False)

def _write_combo_distribution_files(all_results: dict, paths, name_prefix: str,
                                    shortlist_filters: dict = None):
    """
    Build per-method combo distributions and write a single combined CSV.
    Optionally also write a filtered shortlist using quant-style gates.
    """


    frames = []

    # Map keys to user-facing method labels
    method_label = {
        'walk_forward': 'Walk-Forward',
        'expanding':    'Expanding Window',
        'rolling':      'Rolling Window'
    }

    for k, label in method_label.items():
        df = all_results.get(k, pd.DataFrame())
        if df is None or len(df) == 0:
            continue
        dist = _build_combo_distribution(df, label)
        if len(dist):
            frames.append(dist)

            # Save method-specific combo-only distribution file
            if len(dist) > 0:
                combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_distribution_combos_only.csv")
                _drop_heavy_cols(dist).to_csv(combo_path, index=False)

                # Also save method-specific combo-only shortlist
                if shortlist_filters:
                    cf_method = dist.copy()
                    # Apply typical quant-style gates (tune thresholds as needed)
                    # Defaults if key not present
                    sharpe_p50_min = shortlist_filters.get('Sharpe_p50_min', 0.0)
                    return_p10_min = shortlist_filters.get('Return_p10_min', 0.0)
                    sharpe_iqr_max = shortlist_filters.get('Sharpe_IQR_max', np.inf)
                    maxdd_p90_min  = shortlist_filters.get('MaxDD_p90_min', -np.inf)  # closer to 0 is better
                    hitrate_min    = shortlist_filters.get('HitRate_Positive_Sharpe_min', 0.6)

                    cf_method = cf_method[
                        (cf_method['Sharpe_p50'] >= sharpe_p50_min) &
                        (cf_method['Return_p10'] >= return_p10_min) &
                        (cf_method['Sharpe_IQR'] <= sharpe_iqr_max) &
                        (cf_method['MaxDD_p90'] >= maxdd_p90_min) &
                        (cf_method['HitRate_Positive_Sharpe'] >= hitrate_min)
                    ].copy()

                    # Re-rank the shortlist with same rule
                    cf_method = cf_method.sort_values(
                        by=['Sharpe_p50','Return_p10','MaxDD_p90'],
                        ascending=[False, True, False]
                    ).reset_index(drop=True)

                    if len(cf_method) > 0:
                        combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_combos_only.csv")
                        _drop_heavy_cols(cf_method).to_csv(combo_path, index=False)

    if not frames:
        print("No combo rows found across methods; no combo distribution file written.")
        return None, None

    combo_all = pd.concat(frames, ignore_index=True)
    combo_all_path = save_df(combo_all, paths.aggregates, "quant_desk_summary.csv",
                             kind="quant_desk_summary", method="all")

    # Save combo-only quant desk summary
    if len(combo_all) > 0:
        combo_path = os.path.join(paths.aggregates, f"{name_prefix}_quant_desk_summary_combos_only.csv")
        _drop_heavy_cols(combo_all).to_csv(combo_path, index=False)

        # Also save combo-only distribution files for each method
        for k, label in method_label.items():
            if k in all_results and len(all_results[k]) > 0:
                combo_df = all_results[k][has_plus(all_results[k]['Signal'])]
                if len(combo_df) > 0:
                    combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_distribution_combos_only.csv")
                    _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

                    # Also save method-specific combo-only distribution with basic stats
                    if len(combo_df) > 0:
                        # Group by Signal and Ticker to get basic stats
                        g = combo_df.groupby(['Signal','Ticker'], as_index=False)
                        basic_stats = g.agg({
                            'Smart Sharpe': ['mean', 'median', 'std', 'min', 'max'],
                            'Total Return': ['mean', 'median', 'std', 'min', 'max'],
                            'Max Drawdown': ['mean', 'median', 'std', 'min', 'max'],
                            'Time in Market': ['mean', 'median']
                        })

                        # Flatten MultiIndex columns
                        basic_stats.columns = ['_'.join([c for c in col if c]).strip('_') for col in basic_stats.columns.values]

                        # Add basic stats to combo-only file
                        combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_basic_stats_combos_only.csv")
                        _drop_heavy_cols(basic_stats).to_csv(combo_path, index=False)

                        # Also save method-specific combo-only distribution with percentiles
                        if len(combo_df) > 0:
                            # Group by Signal and Ticker to get percentiles
                            g = combo_df.groupby(['Signal','Ticker'], as_index=False)
                            percentiles = g.agg({
                                'Smart Sharpe': [lambda x: np.nanpercentile(x, 10), lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 50), lambda x: np.nanpercentile(x, 75), lambda x: np.nanpercentile(x, 90)],
                                'Total Return': [lambda x: np.nanpercentile(x, 10), lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 50), lambda x: np.nanpercentile(x, 75), lambda x: np.nanpercentile(x, 90)],
                                'Max Drawdown': [lambda x: np.nanpercentile(x, 10), lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 50), lambda x: np.nanpercentile(x, 75), lambda x: np.nanpercentile(x, 90)]
                            })

                            # Flatten MultiIndex columns
                            percentiles.columns = ['_'.join([c for c in col if c]).strip('_') for col in percentiles.columns.values]

                            # Add percentiles to combo-only file
                            combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_percentiles_combos_only.csv")
                            _drop_heavy_cols(percentiles).to_csv(combo_path, index=False)

                            # Also save method-specific combo-only distribution with hit rates
                            if len(combo_df) > 0:
                                # Group by Signal and Ticker to get hit rates
                                g = combo_df.groupby(['Signal','Ticker'], as_index=False)
                                hit_rates = g.agg({
                                    'Smart Sharpe': [lambda x: np.mean(np.array(x) > 0)],
                                    'Total Return': [lambda x: np.mean(np.array(x) > 0)],
                                    'Max Drawdown': [lambda x: np.mean(np.array(x) > -0.5)]  # Consider drawdowns better than -50% as "hits"
                                })

                                # Flatten MultiIndex columns
                                hit_rates.columns = ['_'.join([c for c in col if c]).strip('_') for col in hit_rates.columns.values]

                                # Add hit rates to combo-only file
                                combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_hit_rates_combos_only.csv")
                                _drop_heavy_cols(hit_rates).to_csv(combo_path, index=False)

                                # Also save method-specific combo-only distribution with stability metrics
                                if len(combo_df) > 0:
                                    # Group by Signal and Ticker to get stability metrics
                                    g = combo_df.groupby(['Signal','Ticker'], as_index=False)
                                    stability = g.agg({
                                        'Smart Sharpe': [lambda x: np.std(x) / (np.mean(x) + 1e-12), lambda x: np.max(x) - np.min(x)],
                                        'Total Return': [lambda x: np.std(x) / (np.mean(x) + 1e-12), lambda x: np.max(x) - np.min(x)],
                                        'Max Drawdown': [lambda x: np.std(x) / (np.mean(x) + 1e-12), lambda x: np.max(x) - np.min(x)]
                                    })

                                    # Flatten MultiIndex columns
                                    stability.columns = ['_'.join([c for c in col if c]).strip('_') for col in stability.columns.values]

                                    # Add stability metrics to combo-only file
                                    combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_stability_combos_only.csv")
                                    _drop_heavy_cols(stability).to_csv(combo_path, index=False)

                                    # Also save method-specific combo-only distribution with coverage metrics
                                    if len(combo_df) > 0:
                                        # Group by Signal and Ticker to get coverage metrics
                                        g = combo_df.groupby(['Signal','Ticker'], as_index=False)
                                        coverage = g.agg({
                                            'Smart Sharpe': ['count'],
                                            'Total Return': ['count'],
                                            'Max Drawdown': ['count']
                                        })

                                        # Flatten MultiIndex columns
                                        coverage.columns = ['_'.join([c for c in col if c]).strip('_') for col in coverage.columns.values]

                                        # Add coverage metrics to combo-only file
                                        combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_combo_coverage_combos_only.csv")
                                        _drop_heavy_cols(coverage).to_csv(combo_path, index=False)

    shortlist_path = None
    if shortlist_filters:
        cf = combo_all.copy()
        # Apply typical quant-style gates (tune thresholds as needed)
        # Defaults if key not present
        sharpe_p50_min = shortlist_filters.get('Sharpe_p50_min', 0.0)
        return_p10_min = shortlist_filters.get('Return_p10_min', 0.0)
        sharpe_iqr_max = shortlist_filters.get('Sharpe_IQR_max', np.inf)
        maxdd_p90_min  = shortlist_filters.get('MaxDD_p90_min', -np.inf)  # closer to 0 is better
        hitrate_min    = shortlist_filters.get('HitRate_Positive_Sharpe_min', 0.6)

        cf = cf[
            (cf['Sharpe_p50'] >= sharpe_p50_min) &
            (cf['Return_p10'] >= return_p10_min) &
            (cf['Sharpe_IQR'] <= sharpe_iqr_max) &
            (cf['MaxDD_p90'] >= maxdd_p90_min) &
            (cf['HitRate_Positive_Sharpe'] >= hitrate_min)
        ].copy()

        # Re-rank the shortlist with same rule
        cf = cf.sort_values(
            by=['Sharpe_p50','Return_p10','MaxDD_p90'],
            ascending=[False, True, False]
        ).reset_index(drop=True)

        shortlist_path = save_df(cf, paths.aggregates, "quant_desk_shortlist.csv",
                                kind="quant_desk_shortlist", method="all")

        # Save combo-only shortlist
        if len(cf) > 0:
            combo_path = os.path.join(paths.aggregates, f"{name_prefix}_quant_desk_shortlist_combos_only.csv")
            _drop_heavy_cols(cf).to_csv(combo_path, index=False)

            # Also save combo-only shortlist for each method
            for k, label in method_label.items():
                if k in all_results and len(all_results[k]) > 0:
                    combo_df = all_results[k][has_plus(all_results[k]['Signal'])]
                    if len(combo_df) > 0:
                        combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_combos_only.csv")
                        _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

                        # Also save method-specific combo-only shortlist with filters
                        if shortlist_filters:
                            cf_method = combo_df.copy()
                            # Apply typical quant-style gates (tune thresholds as needed)
                            # Defaults if key not present
                            sharpe_p50_min = shortlist_filters.get('Sharpe_p50_min', 0.0)
                            return_p10_min = shortlist_filters.get('Return_p10_min', 0.0)
                            sharpe_iqr_max = shortlist_filters.get('Sharpe_IQR_max', np.inf)
                            maxdd_p90_min  = shortlist_filters.get('MaxDD_p90_min', -np.inf)  # closer to 0 is better
                            hitrate_min    = shortlist_filters.get('HitRate_Positive_Sharpe_min', 0.6)

                            cf_method = cf_method[
                                (cf_method['Smart Sharpe'] >= sharpe_p50_min) &
                                (cf_method['Total Return'] >= return_p10_min) &
                                (cf_method['Max Drawdown'] >= maxdd_p90_min)
                            ].copy()

                            # Re-rank the shortlist with same rule
                            cf_method = cf_method.sort_values(
                                by=['Smart Sharpe','Total Return','Max Drawdown'],
                                ascending=[False, True, False]
                            ).reset_index(drop=True)

                            if len(cf_method) > 0:
                                combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_filtered_combos_only.csv")
                                _drop_heavy_cols(cf_method).to_csv(combo_path, index=False)

                                # Also save method-specific combo-only shortlist with basic stats
                                if len(cf_method) > 0:
                                    # Group by Signal and Ticker to get basic stats
                                    g = cf_method.groupby(['Signal','Ticker'], as_index=False)
                                    basic_stats = g.agg({
                                        'Smart Sharpe': ['mean', 'median', 'std', 'min', 'max'],
                                        'Total Return': ['mean', 'median', 'std', 'min', 'max'],
                                        'Max Drawdown': ['mean', 'median', 'std', 'min', 'max'],
                                        'Time in Market': ['mean', 'median']
                                    })

                                    # Flatten MultiIndex columns
                                    basic_stats.columns = ['_'.join([c for c in col if c]).strip('_') for col in basic_stats.columns.values]

                                    # Add basic stats to combo-only file
                                    combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_filtered_basic_stats_combos_only.csv")
                                    _drop_heavy_cols(basic_stats).to_csv(combo_path, index=False)

                                    # Also save method-specific combo-only shortlist with percentiles
                                    if len(cf_method) > 0:
                                        # Group by Signal and Ticker to get percentiles
                                        g = cf_method.groupby(['Signal','Ticker'], as_index=False)
                                        percentiles = g.agg({
                                            'Smart Sharpe': [lambda x: np.nanpercentile(x, 10), lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 50), lambda x: np.nanpercentile(x, 75), lambda x: np.nanpercentile(x, 90)],
                                            'Total Return': [lambda x: np.nanpercentile(x, 10), lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 50), lambda x: np.nanpercentile(x, 75), lambda x: np.nanpercentile(x, 90)],
                                            'Max Drawdown': [lambda x: np.nanpercentile(x, 10), lambda x: np.nanpercentile(x, 25), lambda x: np.nanpercentile(x, 50), lambda x: np.nanpercentile(x, 75), lambda x: np.nanpercentile(x, 90)]
                                        })

                                        # Flatten MultiIndex columns
                                        percentiles.columns = ['_'.join([c for c in col if c]).strip('_') for col in percentiles.columns.values]

                                        # Add percentiles to combo-only file
                                        combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_filtered_percentiles_combos_only.csv")
                                        _drop_heavy_cols(percentiles).to_csv(combo_path, index=False)

                                        # Also save method-specific combo-only shortlist with hit rates
                                        if len(cf_method) > 0:
                                            # Group by Signal and Ticker to get hit rates
                                            g = cf_method.groupby(['Signal','Ticker'], as_index=False)
                                            hit_rates = g.agg({
                                                'Smart Sharpe': [lambda x: np.mean(np.array(x) > 0)],
                                                'Total Return': [lambda x: np.mean(np.array(x) > 0)],
                                                'Max Drawdown': [lambda x: np.mean(np.array(x) > -0.5)]  # Consider drawdowns better than -50% as "hits"
                                            })

                                            # Flatten MultiIndex columns
                                            hit_rates.columns = ['_'.join([c for c in col if c]).strip('_') for col in hit_rates.columns.values]

                                            # Add hit rates to combo-only file
                                            combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_filtered_hit_rates_combos_only.csv")
                                            _drop_heavy_cols(hit_rates).to_csv(combo_path, index=False)

                                            # Also save method-specific combo-only shortlist with stability metrics
                                            if len(cf_method) > 0:
                                                # Group by Signal and Ticker to get stability metrics
                                                g = cf_method.groupby(['Signal','Ticker'], as_index=False)
                                                stability = g.agg({
                                                    'Smart Sharpe': [lambda x: np.std(x) / (np.mean(x) + 1e-12), lambda x: np.max(x) - np.min(x)],
                                                    'Total Return': [lambda x: np.std(x) / (np.mean(x) + 1e-12), lambda x: np.max(x) - np.min(x)],
                                                    'Max Drawdown': [lambda x: np.std(x) / (np.mean(x) + 1e-12), lambda x: np.max(x) - np.min(x)]
                                                })

                                                # Flatten MultiIndex columns
                                                stability.columns = ['_'.join([c for c in col if c]).strip('_') for col in stability.columns.values]

                                                # Add stability metrics to combo-only file
                                                combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_filtered_stability_combos_only.csv")
                                                _drop_heavy_cols(stability).to_csv(combo_path, index=False)

                                                # Also save method-specific combo-only shortlist with coverage metrics
                                                if len(cf_method) > 0:
                                                    # Group by Signal and Ticker to get coverage metrics
                                                    g = cf_method.groupby(['Signal','Ticker'], as_index=False)
                                                    coverage = g.agg({
                                                        'Smart Sharpe': ['count'],
                                                        'Total Return': ['count'],
                                                        'Max Drawdown': ['count']
                                                    })

                                                    # Flatten MultiIndex columns
                                                    coverage.columns = ['_'.join([c for c in col if c]).strip('_') for col in coverage.columns.values]

                                                    # Add coverage metrics to combo-only file
                                                    combo_path = os.path.join(paths.aggregates, f"{name_prefix}_{k}_quant_desk_shortlist_filtered_coverage_combos_only.csv")
                                                    _drop_heavy_cols(coverage).to_csv(combo_path, index=False)

    return combo_all_path, shortlist_path

class EvalMode(Enum):
    HOLDOUT_70_30 = "holdout_70_30"
    WALK_FORWARD = "walk_forward"
    EXPANDING = "expanding"
    ROLLING = "rolling"

class EvaluationConfig:
    """Configuration for different evaluation methods"""
    def __init__(self):
        # Holdout configuration
        self.holdout_train_pct = 0.70
        self.embargo_days = 5  # Gap between train/test to avoid leakage

        # Walk-forward configuration
        self.wf_train_period = 252  # Training window size
        self.wf_test_period = 63   # Test window size
        self.wf_step_size = 21     # Step size between windows

        # Expanding window configuration
        self.exp_initial_train = 252  # Initial training size
        self.exp_test_period = 63     # Test period size
        self.exp_expansion_size = 63  # How much to expand each iteration

        # Rolling window configuration
        self.roll_train_period = 252  # Fixed training window size
        self.roll_test_period = 63    # Test period size
        self.roll_step_size = 21      # Step size

        # Monte Carlo configuration
        self.mc_num_simulations = 10000
        self.mc_annual_periods = 252

        # General configuration
        self.min_train_days = 60      # Minimum training data required
        self.robustness_cutoff = 0.5  # Minimum robustness score

def calculate_embargo_split(price_data, train_pct, embargo_days=0, valid_on=None):
    """
    Split using *valid* bars (non-NaN rows) instead of raw row count.
    Returns: train_data, test_data, embargo_start_idx, embargo_end_idx
    """
    df = price_data

    # Determine valid rows
    if isinstance(valid_on, str):
        valid_mask = df[valid_on].notna()
    elif isinstance(valid_on, (list, tuple)) and len(valid_on) > 0:
        valid_mask = df[valid_on].notna().any(axis=1)
    else:
        valid_mask = df.notna().any(axis=1)

    # Get indices of valid rows
    valid_indices = df.index[valid_mask]

    if len(valid_indices) == 0:
        print("No valid rows to split.")
        return df.iloc[:0], df.iloc[:0], 0, 0

    # Calculate split position based on valid bars
    num_valid_bars = len(valid_indices)
    train_valid_bars = int(num_valid_bars * float(train_pct))
    train_valid_bars = max(1, min(train_valid_bars, num_valid_bars - 1))

    # Get the actual date for the split (last date in training)
    split_date = valid_indices[train_valid_bars - 1]

    # Convert back to DataFrame positions
    split_iloc = df.index.get_loc(split_date)

    # Calculate embargo period in calendar days
    embargo_start_idx = split_iloc + 1
    embargo_end_idx = min(embargo_start_idx + embargo_days, len(df))

    # Create splits
    train_data = df.iloc[:split_iloc + 1]
    test_data = df.iloc[embargo_end_idx:]

    # Validation and reporting
    train_valid_count = valid_mask.iloc[:split_iloc + 1].sum()
    test_valid_count = valid_mask.iloc[embargo_end_idx:].sum()

    print(f"Data split with embargo (valid-bar aware):")
    print(f"  Training: {train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}")
    print(f"    Calendar days: {len(train_data)}, Valid bars: {train_valid_count}")
    print(f"    Target valid bars: {train_valid_bars} ({train_pct:.1%})")

    if embargo_days > 0 and embargo_end_idx > embargo_start_idx:
        embargo_start_date = df.index[embargo_start_idx].strftime('%Y-%m-%d')
        embargo_end_date = df.index[embargo_end_idx - 1].strftime('%Y-%m-%d')
        print(f"  Embargo: {embargo_start_date} to {embargo_end_date} ({embargo_days} calendar days)")

    if len(test_data) == 0:
        print("  Testing: <empty> - increase history or reduce embargo.")
    else:
        print(f"  Testing: {test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}")
        print(f"    Calendar days: {len(test_data)}, Valid bars: {test_valid_count}")

    # Verify the split ratio
    actual_train_pct = train_valid_count / (train_valid_count + test_valid_count) if (train_valid_count + test_valid_count) > 0 else 0
    print(f"  Actual train/test split: {actual_train_pct:.1%} / {1-actual_train_pct:.1%}")

    return train_data, test_data, embargo_start_idx, embargo_end_idx

def run_monte_carlo_validation(returns, num_simulations=10000, simulation_length=None, annual_periods=252, random_state=None):
    """
    Run Monte Carlo simulation using simple returns in decimal form
    """
    if simulation_length is None:
        simulation_length = len(returns)

    r = np.asarray(returns, dtype=float)
    # r are simple returns in decimals

    # Remove NaN and infinite values
    r = r[~np.isnan(r)]
    r = r[~np.isinf(r)]

    if len(r) == 0:
        return None

    pos = r[r > 0]
    neg = r[r <= 0]

    if len(pos) == 0 or len(neg) == 0:
        print("Warning: No positive or negative returns for Monte Carlo simulation")
        return None

    ppos = len(pos) / len(r)

    paths = np.zeros((num_simulations, simulation_length + 1))
    mdd = np.zeros(num_simulations)
    final = np.zeros(num_simulations)

    rng = np.random.default_rng(random_state)
    for i in _tqdm(range(num_simulations), total=num_simulations,
                   desc=f"MC sims ({simulation_length} bars)", leave=False):
        sim = np.where(rng.random(simulation_length) < ppos,
                       rng.choice(pos, size=simulation_length, replace=True),
                       rng.choice(neg, size=simulation_length, replace=True))
        eq = np.cumprod(1 + sim)
        paths[i, 1:] = eq - 1
        peak = np.maximum.accumulate(eq)
        dd = 1 - eq / peak
        mdd[i] = np.nanmax(dd)
        final[i] = eq[-1] - 1

    pct = {
        '5':  np.percentile(paths, 5,  axis=0),
        '25': np.percentile(paths, 25, axis=0),
        '50': np.percentile(paths, 50, axis=0),
        '75': np.percentile(paths, 75, axis=0),
        '95': np.percentile(paths, 95, axis=0),
    }
    return {'final_returns': final, 'max_drawdowns': mdd, 'paths': paths, 'percentiles': pct}

def evaluate_signal_performance(signal_returns, benchmark_returns=None, mode="OOS", period_name="", config=None):
    """
    Comprehensive signal evaluation with Monte Carlo validation
    """
    returns = _sanitize_returns(signal_returns)
    if len(returns) == 0:
        return None

    # Calculate basic metrics using quantstats (returns are already simple returns)
    metrics = calculate_quantstats_metrics(returns, benchmark_returns)

    # Add Monte Carlo validation for periods >= 20 days
    if len(returns) >= 20:
        try:
            # Use config if available, otherwise default to 1000
            mc_sims = config.mc_num_simulations if config and hasattr(config, 'mc_num_simulations') else 1000
            # Create deterministic seed from signal name and period for reproducibility
            seed = hash(f"{period_name}_{len(returns)}") % (2**32) if period_name else None
            mc_results = run_monte_carlo_validation(returns, num_simulations=mc_sims, simulation_length=len(returns), random_state=seed)

            if mc_results is not None:
                # Calculate actual final return for comparison (simple compounding, no scaling)
                actual_final = (1 + pd.Series(returns)).prod() - 1

                # Get percentile of actual performance
                percentile_rank = stats.percentileofscore(mc_results['final_returns'], actual_final)

                # Add Monte Carlo metrics
                metrics.update({
                    f'{mode}_MC_Percentile': percentile_rank,
                    f'{mode}_MC_Median_Return': mc_results['percentiles']['50'][-1],
                    f'{mode}_MC_P5_Return': mc_results['percentiles']['5'][-1],
                    f'{mode}_MC_P95_Return': mc_results['percentiles']['95'][-1],
                    f'{mode}_Expected_Max_DD': np.mean(mc_results['max_drawdowns']),
                    f'{mode}_MC_Coverage_90': 1 if mc_results['percentiles']['5'][-1] <= actual_final <= mc_results['percentiles']['95'][-1] else 0,
                    f'{mode}_MC_Coverage_50': 1 if mc_results['percentiles']['25'][-1] <= actual_final <= mc_results['percentiles']['75'][-1] else 0
                })
            else:
                # Add default MC metrics if simulation failed
                for key in [f'{mode}_MC_Percentile', f'{mode}_MC_Median_Return', f'{mode}_MC_Coverage_90', f'{mode}_MC_Coverage_50']:
                    metrics[key] = np.nan

        except Exception as e:
            print(f"Warning: Monte Carlo validation failed for {period_name}: {e}")
            # Add default MC metrics
            for key in [f'{mode}_MC_Percentile', f'{mode}_MC_Median_Return', f'{mode}_MC_Coverage_90', f'{mode}_MC_Coverage_50']:
                metrics[key] = np.nan

    return metrics

# === FULLY CORRECTED AND FEATURE-COMPLETE VERSION of run_walk_forward_evaluation ===
def run_walk_forward_evaluation(signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None, precond_mask=None):
    """
    Run walk-forward analysis on signals.
    Includes full checkpointing, ETA, and optimization logic.
    """
    print(f"\n=== WALK-FORWARD EVALUATION ===")
    total_days = len(price_data)

    # Compute windows until both train and test windows meet minimum sizes
    min_test = 10
    i = 0
    while True:
        train_start = i * config.wf_step_size
        train_end   = train_start + config.wf_train_period
        test_start  = train_end
        test_end    = test_start + config.wf_test_period
        if test_end > total_days or (test_end - test_start) < min_test:
            break
        i += 1

    num_iterations = i
    # num_iterations = min(num_iterations, 50) # Cap at 50 iterations (Disabled)
    print(f"Running {num_iterations} walk-forward iterations")
    print(f"Training window: {config.wf_train_period} days")
    print(f"Test window: {config.wf_test_period} days")
    print(f"Step size: {config.wf_step_size} days")

    checkpoint_path = os.path.join(paths.iter_dir("walk_forward"), "checkpoint.pkl")

    results = []
    start_iteration = 0

    print(f"DIAGNOSTIC H3: Checking for granular walk-forward checkpoint at path: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        print("DIAGNOSTIC H3: Granular checkpoint FOUND. Attempting to load...")
        try:
            with open(checkpoint_path, 'rb') as f:
                progress = pickle.load(f)
            
            results = progress['results'] 
            start_iteration = progress['last_completed_iteration'] + 1 
            
            print(f"✓ DIAGNOSTIC H3: Successfully loaded progress. Resuming walk-forward from iteration {start_iteration + 1}/{num_iterations}...")
        except Exception as e:
            print(f"  ✗ DIAGNOSTIC H3: WARNING: Could not load checkpoint file. Error: {e}. Starting fresh.")
            results = []
            start_iteration = 0
    else:
        print("DIAGNOSTIC H3: Granular checkpoint NOT FOUND. Starting walk-forward from iteration 0.")

    loop_start_time = datetime.now()
    last_update_time = loop_start_time

    for i in _tqdm(range(start_iteration, num_iterations), desc="Walk-Forward windows", leave=False):
        train_start = i * config.wf_step_size
        train_end   = train_start + config.wf_train_period
        test_start  = train_end
        test_end    = test_start + config.wf_test_period

        train_data = price_data.iloc[train_start:train_end]
        test_data  = price_data.iloc[test_start:test_end]

        train_period = f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}"
        test_period  = f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}"

        if i > 0 and i % 5 == 0:  # Update every 5 iterations
            now = datetime.now()
            total_elapsed_time = now - loop_start_time
            total_elapsed_str = str(total_elapsed_time).split('.')[0]
            time_since_last_update = now - last_update_time
            recent_avg_per_iter = time_since_last_update / 5
            remaining_iters = num_iterations - (i + 1)
            eta = now + (recent_avg_per_iter * remaining_iters)
            
            safe_print(f"\nIteration {i+1}/{num_iterations}: Train {train_period}, Test {test_period}")
            safe_print(f"  [Progress] Total Elapsed: {total_elapsed_str}. Recent Avg: ~{str(recent_avg_per_iter).split('.')[0]} per iter.")
            safe_print(f"  [Progress] ACCURATE Estimated Finish Time: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

            # === CHECKPOINTING STEP 4: SAVE THE CURRENT PROGRESS ===
            try:
                progress_data = {'last_completed_iteration': i, 'results': results}
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(progress_data, f)
                safe_print("  [Checkpoint] Progress saved successfully.")
            except Exception as e:
                safe_print(f"  [Checkpoint] WARNING: Could not save checkpoint. Error: {e}")
            
            last_update_time = now

        train_results = backtest_signals(signals, train_data, target_tickers, period_name="train", precond_mask=precond_mask)
        train_results.drop(columns=['Signal Returns'], inplace=True, errors='ignore')

        test_results  = backtest_signals(signals, test_data,  target_tickers, period_name="test", precond_mask=precond_mask)

        combo_df = pd.DataFrame()
        if base_cfg and base_cfg.get('enable_synergistic_combos', False):
            combo_df = enrich_with_synergistic_combos(
                signals=signals,
                train_data=train_data,
                test_data=test_data,
                target_tickers=target_tickers,
                train_results=train_results,
                test_results=test_results,
                sort_by='Smart Sharpe',
                K_primary=min(20, base_cfg.get('k_primary', 30)),
                M_partner=min(30, base_cfg.get('m_partner', 40)),
                ops=("AND", "A_AND_NOT_B", "B_AND_NOT_A", "OR"),
                min_train_gain=base_cfg.get('min_train_gain', 0.05),
                min_test_gain=base_cfg.get('min_test_gain', 0.00),
                max_legs=base_cfg.get('max_combo_legs', 2)
            )

        if len(combo_df) > 0:
            iteration_log_dir = os.path.join(paths.iter_dir("walk_forward"), "iteration_combo_logs")
            os.makedirs(iteration_log_dir, exist_ok=True)
            
            save_df(combo_df, iteration_log_dir,
                    f"combos_iter{(i+1):02d}.csv",
                    kind="iter_combos_log", method="walk_forward",
                    note=f"WF iteration {i+1}", prefix="02")

            combo_path = os.path.join(paths.iter_dir("walk_forward"), f"{name_prefix}_wf_iter{(i+1):02d}_combos_only.csv")
            _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

            train_cols = ['Signal', 'Ticker', 'Train_Smart_Sharpe', 'Train_Sharpe_Ratio', 'Train_Sortino_Ratio', 'Train_Calmar_Ratio', 'Train_Max_Drawdown', 'Train_Total_Return']
            train_rename = {'Train_Smart_Sharpe': 'Smart Sharpe', 'Train_Sharpe_Ratio': 'Sharpe Ratio', 'Train_Sortino_Ratio': 'Sortino Ratio', 'Train_Calmar_Ratio': 'Calmar Ratio', 'Train_Max_Drawdown': 'Max Drawdown', 'Train_Total_Return': 'Total Return'}
            test_cols = ['Signal', 'Ticker', 'Smart Sharpe', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Signal Returns']
            train_results = pd.concat([train_results, combo_df[train_cols].rename(columns=train_rename)], ignore_index=True)
            test_results = pd.concat([test_results, combo_df[test_cols]], ignore_index=True)

        merged, _ = merge_train_test_results(train_results, test_results, 'Smart Sharpe')
        if len(merged) > 0:
            merged = merged.copy()
            merged['WF_Iteration'] = i + 1
            merged['Train_Period'] = train_period
            merged['Test_Period']  = test_period
            merged['Train_Days']   = len(train_data)
            merged['Test_Days']    = len(test_data)
            results.append(merged)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        save_df(all_results, paths.walk_forward, "combos_and_solos.csv", kind="combos_and_solos", method="walk_forward", note="all WF iterations combined", prefix="01")
        save_df(all_results, paths.walk_forward, "results.csv", kind="results", method="walk_forward", note="all WF iterations combined", prefix="00")

        if len(all_results) > 0:
            combo_df = all_results[has_plus(all_results['Signal'])]
            if len(combo_df) > 0:
                combo_path = os.path.join(paths.walk_forward, f"{name_prefix}_walk_forward_combos_only.csv")
                _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print("\n--- Walk-Forward run complete. Checkpoint file removed. ---")
            except Exception as e:
                print(f"\n--- Walk-Forward run complete. Warning: could not remove checkpoint. Error: {e} ---")

        return all_results

    return pd.DataFrame()

# === FINAL CORRECTED VERSION of run_rolling_window_evaluation ===
def run_rolling_window_evaluation(signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None, precond_mask=None):
    """
    Run rolling window analysis with fixed-size training windows.
    Includes full checkpointing, ETA, and optimization logic.
    """
    print(f"\n=== ROLLING WINDOW EVALUATION ===")

    total_days = len(price_data)
    
    if total_days < (config.roll_train_period + config.roll_test_period):
        print(f"Not enough data for rolling window test")
        return pd.DataFrame()

    min_test = 10
    i = 0
    while True:
        train_start = i * config.roll_step_size
        train_end = train_start + config.roll_train_period
        test_start = train_end
        test_end = test_start + config.roll_test_period
        if test_end > total_days or (test_end - test_start) < min_test or train_end >= total_days:
            break
        i += 1

    num_iterations = i
    print(f"Running {num_iterations} rolling window iterations")
    print(f"Training window: {config.roll_train_period} days (fixed)")
    print(f"Test period: {config.roll_test_period} days")
    print(f"Step size: {config.roll_step_size} days")

    checkpoint_path = os.path.join(paths.iter_dir("rolling"), "checkpoint.pkl")

    results = []
    start_iteration = 0

    print(f"DIAGNOSTIC H3: Checking for granular rolling checkpoint at path: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        print("DIAGNOSTIC H3: Granular checkpoint FOUND. Attempting to load...")
        try:
            with open(checkpoint_path, 'rb') as f:
                progress = pickle.load(f)
            results = progress['results'] 
            start_iteration = progress['last_completed_iteration'] + 1 
            print(f"✓ DIAGNOSTIC H3: Successfully loaded progress. Resuming rolling window from iteration {start_iteration + 1}/{num_iterations}...")
        except Exception as e:
            print(f"  ✗ DIAGNOSTIC H3: WARNING: Could not load checkpoint file. Error: {e}. Starting fresh.")
            results = []
            start_iteration = 0
    else:
        print("DIAGNOSTIC H3: Granular checkpoint NOT FOUND. Starting rolling window from iteration 0.")

    loop_start_time = datetime.now()
    last_update_time = loop_start_time

    for i in range(start_iteration, num_iterations):
        train_start = i * config.roll_step_size
        train_end = train_start + config.roll_train_period
        test_start = train_end
        test_end = test_start + config.roll_test_period

        train_data = price_data.iloc[train_start:train_end]
        test_data = price_data.iloc[test_start:test_end]

        train_period = f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}"
        test_period = f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}"

        if i > 0 and i % 5 == 0:
            now = datetime.now()
            total_elapsed_time = now - loop_start_time
            total_elapsed_str = str(total_elapsed_time).split('.')[0]
            time_since_last_update = now - last_update_time
            recent_avg_per_iter = time_since_last_update / 5
            remaining_iters = num_iterations - (i + 1)
            eta = now + (recent_avg_per_iter * remaining_iters)
            
            safe_print(f"\nIteration {i+1}/{num_iterations}: Train {train_period}, Test {test_period}")
            safe_print(f"  [Progress] Total Elapsed: {total_elapsed_str}. Recent Avg: ~{str(recent_avg_per_iter).split('.')[0]} per iter.")
            safe_print(f"  [Progress] ACCURATE Estimated Finish Time: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                progress_data = {'last_completed_iteration': i, 'results': results}
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(progress_data, f)
                safe_print("  [Checkpoint] Progress saved successfully.")
            except Exception as e:
                safe_print(f"  [Checkpoint] WARNING: Could not save checkpoint. Error: {e}")

            last_update_time = now

        train_results = backtest_signals(signals, train_data, target_tickers, period_name="train", precond_mask=precond_mask)
        train_results.drop(columns=['Signal Returns'], inplace=True, errors='ignore')

        test_results = backtest_signals(signals, test_data, target_tickers, period_name="test", precond_mask=precond_mask)

        combo_df = pd.DataFrame()
        if base_cfg and base_cfg.get('enable_synergistic_combos', False):
            combo_df = enrich_with_synergistic_combos(
                signals=signals, train_data=train_data, test_data=test_data,
                target_tickers=target_tickers, train_results=train_results,
                test_results=test_results, sort_by='Smart Sharpe',
                K_primary=min(20, base_cfg.get('k_primary', 30)),
                M_partner=min(30, base_cfg.get('m_partner', 40)),
                ops=("AND", "A_AND_NOT_B", "B_AND_NOT_A", "OR"),
                min_train_gain=base_cfg.get('min_train_gain', 0.05),
                min_test_gain=base_cfg.get('min_test_gain', 0.00),
                max_legs=base_cfg.get('max_combo_legs', 2)
            )

        if len(combo_df) > 0:
            iteration_log_dir = os.path.join(paths.iter_dir("rolling"), "iteration_combo_logs")
            os.makedirs(iteration_log_dir, exist_ok=True)
            save_df(combo_df, iteration_log_dir,
                    f"combos_iter{(i+1):02d}.csv",
                    kind="iter_combos_log", method="rolling",
                    note=f"ROLL iteration {i+1}", prefix="02")

            combo_path = os.path.join(paths.iter_dir("rolling"), f"{name_prefix}_roll_iter{(i+1):02d}_combos_only.csv")
            _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

            train_cols = ['Signal', 'Ticker', 'Train_Smart_Sharpe', 'Train_Sharpe_Ratio', 'Train_Sortino_Ratio', 'Train_Calmar_Ratio', 'Train_Max_Drawdown', 'Train_Total_Return']
            train_rename = {'Train_Smart_Sharpe': 'Smart Sharpe', 'Train_Sharpe_Ratio': 'Sharpe Ratio', 'Train_Sortino_Ratio': 'Sortino Ratio', 'Train_Calmar_Ratio': 'Calmar Ratio', 'Train_Max_Drawdown': 'Max Drawdown', 'Train_Total_Return': 'Total Return'}
            test_cols = ['Signal', 'Ticker', 'Smart Sharpe', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Signal Returns']

            train_results = pd.concat([train_results, combo_df[train_cols].rename(columns=train_rename)], ignore_index=True)
            test_results = pd.concat([test_results, combo_df[test_cols]], ignore_index=True)

        merged, _ = merge_train_test_results(train_results, test_results, 'Smart Sharpe')

        if len(merged) > 0:
            merged = merged.copy()
            merged['Roll_Iteration'] = i + 1
            merged['Train_Period'] = train_period
            merged['Test_Period'] = test_period
            merged['Train_Days'] = len(train_data)
            merged['Test_Days'] = len(test_data)
            results.append(merged)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        save_df(all_results, paths.rolling, "combos_and_solos.csv", kind="combos_and_solos", method="rolling", note="all ROLL iterations combined", prefix="01")
        save_df(all_results, paths.rolling, "results.csv", kind="results", method="rolling", note="all ROLL iterations combined", prefix="00")

        if len(all_results) > 0:
            combo_df = all_results[has_plus(all_results['Signal'])]
            if len(combo_df) > 0:
                combo_path = os.path.join(paths.rolling, f"{name_prefix}_rolling_combos_only.csv")
                _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print("\n--- Rolling Window run complete. Checkpoint file removed. ---")
            except Exception as e:
                print(f"\n--- Rolling Window run complete. Warning: could not remove checkpoint. Error: {e} ---")
        
        return all_results

    return pd.DataFrame()

# === FULLY CORRECTED AND FEATURE-COMPLETE VERSION of run_expanding_window_evaluation ===
def run_expanding_window_evaluation(signals, price_data, target_tickers, config, paths, name_prefix="", base_cfg=None, precond_mask=None):
    """
    Run expanding window analysis on signals.
    Includes full checkpointing, ETA, and optimization logic.
    """
    print(f"\n=== EXPANDING WINDOW EVALUATION ===")

    total_days = len(price_data)

    if total_days < (config.exp_initial_train + config.exp_test_period):
        print(f"Not enough data for expanding window test")
        return pd.DataFrame()

    min_test = 10
    i = 0
    while True:
        train_start = 0
        train_size = config.exp_initial_train + (i * config.exp_expansion_size)
        train_end = train_size
        test_start = train_end
        test_end = test_start + config.exp_test_period
        if test_end > total_days or (test_end - test_start) < min_test or train_end >= total_days:
            break
        i += 1

    num_iterations = i
    print(f"Running {num_iterations} expanding window iterations")
    print(f"Initial training: {config.exp_initial_train} days")
    print(f"Test period: {config.exp_test_period} days")
    print(f"Expansion size: {config.exp_expansion_size} days")

    checkpoint_path = os.path.join(paths.iter_dir("expanding"), "checkpoint.pkl")

    results = []
    start_iteration = 0

    print(f"DIAGNOSTIC H3: Checking for granular expanding checkpoint at path: {checkpoint_path}")
    if os.path.exists(checkpoint_path):
        print("DIAGNOSTIC H3: Granular checkpoint FOUND. Attempting to load...")
        try:
            with open(checkpoint_path, 'rb') as f:
                progress = pickle.load(f)
            results = progress['results'] 
            start_iteration = progress['last_completed_iteration'] + 1 
            print(f"✓ DIAGNOSTIC H3: Successfully loaded progress. Resuming expanding window from iteration {start_iteration + 1}/{num_iterations}...")
        except Exception as e:
            print(f"  ✗ DIAGNOSTIC H3: WARNING: Could not load checkpoint file. Error: {e}. Starting fresh.")
            results = []
            start_iteration = 0
    else:
        print("DIAGNOSTIC H3: Granular checkpoint NOT FOUND. Starting expanding window from iteration 0.")

    loop_start_time = datetime.now()
    last_update_time = loop_start_time

    for i in range(start_iteration, num_iterations):
        train_start = 0
        train_size = config.exp_initial_train + (i * config.exp_expansion_size)
        train_end = train_size
        test_start = train_end
        test_end = test_start + config.exp_test_period

        train_data = price_data.iloc[train_start:train_end]
        test_data = price_data.iloc[test_start:test_end]

        train_period = f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}"
        test_period = f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}"

        if i > 0 and i % 5 == 0:
            now = datetime.now()
            total_elapsed_time = now - loop_start_time
            total_elapsed_str = str(total_elapsed_time).split('.')[0]
            time_since_last_update = now - last_update_time
            recent_avg_per_iter = time_since_last_update / 5
            remaining_iters = num_iterations - (i + 1)
            eta = now + (recent_avg_per_iter * remaining_iters)
            
            safe_print(f"\nIteration {i+1}/{num_iterations}: Train {train_period} ({len(train_data)} days), Test {test_period}")
            safe_print(f"  [Progress] Total Elapsed: {total_elapsed_str}. Recent Avg: ~{str(recent_avg_per_iter).split('.')[0]} per iter.")
            safe_print(f"  [Progress] ACCURATE Estimated Finish Time: {eta.strftime('%Y-%m-%d %H:%M:%S')}")
            
            try:
                progress_data = {'last_completed_iteration': i, 'results': results}
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(progress_data, f)
                safe_print("  [Checkpoint] Progress saved successfully.")
            except Exception as e:
                safe_print(f"  [Checkpoint] WARNING: Could not save checkpoint. Error: {e}")

            last_update_time = now

        train_results = backtest_signals(signals, train_data, target_tickers, period_name="train")
        train_results.drop(columns=['Signal Returns'], inplace=True, errors='ignore')
        
        test_results = backtest_signals(signals, test_data, target_tickers, period_name="test")

        combo_df = pd.DataFrame()
        if base_cfg and base_cfg.get('enable_synergistic_combos', False):
            combo_df = enrich_with_synergistic_combos(
                signals=signals, train_data=train_data, test_data=test_data,
                target_tickers=target_tickers, train_results=train_results,
                test_results=test_results, sort_by='Smart Sharpe',
                K_primary=min(20, base_cfg.get('k_primary', 30)),
                M_partner=min(30, base_cfg.get('m_partner', 40)),
                ops=("AND", "A_AND_NOT_B", "B_AND_NOT_A", "OR"),
                min_train_gain=base_cfg.get('min_train_gain', 0.05),
                min_test_gain=base_cfg.get('min_test_gain', 0.00),
                max_legs=base_cfg.get('max_combo_legs', 2)
            )

        if len(combo_df) > 0:
            iteration_log_dir = os.path.join(paths.iter_dir("expanding"), "iteration_combo_logs")
            os.makedirs(iteration_log_dir, exist_ok=True)
            save_df(combo_df, iteration_log_dir,
                    f"combos_iter{(i+1):02d}.csv",
                    kind="iter_combos_log", method="expanding",
                    note=f"EW iteration {i+1}", prefix="02")

            combo_path = os.path.join(paths.iter_dir("expanding"), f"{name_prefix}_ew_iter{(i+1):02d}_combos_only.csv")
            _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

            train_cols = ['Signal', 'Ticker', 'Train_Smart_Sharpe', 'Train_Sharpe_Ratio', 'Train_Sortino_Ratio', 'Train_Calmar_Ratio', 'Train_Max_Drawdown', 'Train_Total_Return']
            train_rename = {'Train_Smart_Sharpe': 'Smart Sharpe', 'Train_Sharpe_Ratio': 'Sharpe Ratio', 'Train_Sortino_Ratio': 'Sortino Ratio', 'Train_Calmar_Ratio': 'Calmar Ratio', 'Train_Max_Drawdown': 'Max Drawdown', 'Train_Total_Return': 'Total Return'}
            test_cols = ['Signal', 'Ticker', 'Smart Sharpe', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Signal Returns']

            train_results = pd.concat([train_results, combo_df[train_cols].rename(columns=train_rename)], ignore_index=True)
            test_results = pd.concat([test_results, combo_df[test_cols]], ignore_index=True)

        merged, _ = merge_train_test_results(train_results, test_results, 'Smart Sharpe')

        if len(merged) > 0:
            merged = merged.copy()
            merged['EW_Iteration'] = i + 1
            merged['Train_Period'] = train_period
            merged['Test_Period'] = test_period
            merged['Train_Days'] = len(train_data)
            merged['Test_Days'] = len(test_data)
            results.append(merged)

    if results:
        all_results = pd.concat(results, ignore_index=True)
        save_df(all_results, paths.expanding, "combos_and_solos.csv", kind="combos_and_solos", method="expanding", note="all EW iterations combined", prefix="01")
        save_df(all_results, paths.expanding, "results.csv", kind="results", method="expanding", note="all EW iterations combined", prefix="00")

        if len(all_results) > 0:
            combo_df = all_results[has_plus(all_results['Signal'])]
            if len(combo_df) > 0:
                combo_path = os.path.join(paths.expanding, f"{name_prefix}_expanding_combos_only.csv")
                _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                print("\n--- Expanding Window run complete. Checkpoint file removed. ---")
            except Exception as e:
                print(f"\n--- Expanding Window run complete. Warning: could not remove checkpoint. Error: {e} ---")

        return all_results

    return pd.DataFrame()

# === FINAL, CORRECTED VERSION of run_comprehensive_evaluation ===
def run_comprehensive_evaluation(signals, price_data, target_tickers, eval_modes, config, paths, name_prefix="", base_cfg=None, completed_evaluations=None):
    """
    Run comprehensive evaluation using multiple methods.
    This version is resume-aware and will skip methods that are already completed.
    It now also RETURNS the updated list of completed evaluations.
    """
    if base_cfg is None:
        base_cfg = {}
    
    # --- SAFETY CHECK for the bookmark ---
    if completed_evaluations is None:
        completed_evaluations = []
    # ------------------------------------

    import pandas as pd
    all_results = {
        'walk_forward': pd.DataFrame(),
        'expanding':    pd.DataFrame(),
        'rolling':      pd.DataFrame(),
        'holdout': {}
    }

    # Build precond_mask (if configured) and pass it through to each evaluation.
    # Masking is applied per-windowed-slice inside _run_single_backtest so it
    # is correct within each walk-forward / rolling / expanding window.
    precond_mask = None
    if base_cfg.get('preconditions'):
        print("\nBuilding precondition mask...")
        try:
            precond_mask = build_precondition_series(
                price_data, base_cfg['preconditions'], base_cfg.get('precondition_combine', 'AND')
            )
            if isinstance(precond_mask, pd.DataFrame):
                precond_mask = precond_mask.iloc[:, 0]
            print("✓ Precondition mask built.")
        except Exception as e:
            print(f"⚠️ Precondition error: {e}. Continuing without precondition mask.")
    else:
        print("\n[preconditions] Active expressions: (none)")


    # --- 1. Holdout evaluation ---
    if EvalMode.HOLDOUT_70_30 in eval_modes and EvalMode.HOLDOUT_70_30 not in completed_evaluations:
        print(f"\n{'='*60}\nRunning HOLDOUT EVALUATION\n{'='*60}")
        
        train_data, test_data, _, _ = calculate_embargo_split(
            price_data, config.holdout_train_pct, config.embargo_days, target_tickers[0])
        
        train_results = backtest_signals(signals, train_data, target_tickers, "train", precond_mask=precond_mask)
        test_results = backtest_signals(signals, test_data, target_tickers, "test", precond_mask=precond_mask)
        
        if base_cfg.get('enable_synergistic_combos', False):
            combo_df = enrich_with_synergistic_combos(
                signals=signals, train_data=train_data, test_data=test_data,
                target_tickers=target_tickers, train_results=train_results,
                test_results=test_results, sort_by='Smart Sharpe',
                K_primary=base_cfg.get('k_primary', 30), M_partner=base_cfg.get('m_partner', 40),
                ops=("AND", "A_AND_NOT_B", "B_AND_NOT_A", "OR"),
                min_train_gain=base_cfg.get('min_train_gain', 0.05),
                min_test_gain=base_cfg.get('min_test_gain', 0.00),
                max_legs=base_cfg.get('max_combo_legs', 2)
            )
            if len(combo_df) > 0:
                save_df(combo_df, paths.holdout, "combos_only.csv",
                        kind="combos_only", method="holdout", note="just-created combos", prefix="01")
                train_cols = ['Signal', 'Ticker', 'Train_Smart_Sharpe', 'Train_Sharpe_Ratio', 'Train_Sortino_Ratio', 'Train_Calmar_Ratio', 'Train_Max_Drawdown', 'Train_Total_Return']
                train_rename = {'Train_Smart_Sharpe': 'Smart Sharpe', 'Train_Sharpe_Ratio': 'Sharpe Ratio', 'Train_Sortino_Ratio': 'Sortino Ratio', 'Train_Calmar_Ratio': 'Calmar Ratio', 'Train_Max_Drawdown': 'Max Drawdown', 'Train_Total_Return': 'Total Return'}
                test_cols = ['Signal', 'Ticker', 'Smart Sharpe', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Max Drawdown', 'Total Return', 'Signal Returns']
                train_results = pd.concat([train_results, combo_df[train_cols].rename(columns=train_rename)], ignore_index=True)
                test_results = pd.concat([test_results, combo_df[test_cols]], ignore_index=True)

        merged_results, robust_results = merge_train_test_results(train_results, test_results, 'Smart Sharpe')
        filtered_results = apply_comprehensive_filter(
            merged_results, tim_min=base_cfg.get('tim', 0.025),
            mdd_max=base_cfg.get('mdd', -0.75), quant_filter=base_cfg.get('quant', 0.66),
            robustness_cutoff=config.robustness_cutoff
        )
        save_df(merged_results, paths.holdout, "combos_and_solos.csv",
                kind="combos_and_solos", method="holdout", note="pre-filter merged results", prefix="02")
        
        if len(filtered_results) > 0:
            save_df(filtered_results, paths.holdout, "results.csv",
                    kind="results", method="holdout", note="solos+combos ranked", prefix="00")

        all_results['holdout'] = {
            'merged_results': merged_results, 'robust_results': robust_results,
            'filtered_results': filtered_results,
            'train_period': f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}",
            'test_period': f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
            'embargo_days': config.embargo_days
        }
        
        completed_evaluations.append(EvalMode.HOLDOUT_70_30)
        print("--- Holdout Evaluation Complete ---")

    # --- 2. Walk-forward evaluation ---
    if EvalMode.WALK_FORWARD in eval_modes and EvalMode.WALK_FORWARD not in completed_evaluations:
        wf_results = run_walk_forward_evaluation(signals, price_data, target_tickers, config, paths, name_prefix, base_cfg, precond_mask=precond_mask)
        all_results['walk_forward'] = wf_results
        completed_evaluations.append(EvalMode.WALK_FORWARD)
        print("--- Walk-Forward Evaluation Complete ---")

    # --- 3. Expanding window evaluation ---
    if EvalMode.EXPANDING in eval_modes and EvalMode.EXPANDING not in completed_evaluations:
        ew_results = run_expanding_window_evaluation(signals, price_data, target_tickers, config, paths, name_prefix, base_cfg, precond_mask=precond_mask)
        all_results['expanding'] = ew_results
        completed_evaluations.append(EvalMode.EXPANDING)
        print("--- Expanding Window Evaluation Complete ---")

    # --- 4. Rolling window evaluation ---
    if EvalMode.ROLLING in eval_modes and EvalMode.ROLLING not in completed_evaluations:
        roll_results = run_rolling_window_evaluation(signals, price_data, target_tickers, config, paths, name_prefix, base_cfg, precond_mask=precond_mask)
        all_results['rolling'] = roll_results
        completed_evaluations.append(EvalMode.ROLLING)
        print("--- Rolling Window Evaluation Complete ---")

    # --- 5. Final Reporting ---
    print("\n--- Generating Final Reports ---")
    generate_evaluation_summary(all_results, paths, name_prefix, top_n_per_method=50)
    save_method_averages(all_results, paths, name_prefix, per_method_files=True)
    export_all_combo_artifacts(
        all_results=all_results, signals=signals, full_prices=price_data,
        output_dir=paths.root, name_prefix=name_prefix,
        frozen_top=base_cfg.get('frozen_top_k', 30),
        corr_threshold=base_cfg.get('combo_corr_threshold', 0.30),
        shortlist_size=base_cfg.get('combo_shortlist_size', 12)
    )
    
    # CRUCIAL: Return both the results AND the updated bookmark list
    return all_results, completed_evaluations

# ---- SMART SHARPE WEIGHT OPTIMIZATION (single-file, plug & play) ----
import numpy as _np
import pandas as _pd
import os as _os
from scipy.optimize import minimize as _minimize
import quantstats as _qs

def _smart_sharpe_safe(series):
    s = _pd.Series(series).replace([_np.inf, -_np.inf], _np.nan).fillna(0.0)
    if s.std(ddof=0) == 0 or s.shape[0] < 5:
        return -_np.inf
    try:
        val = _qs.stats.smart_sharpe(s)
        if _np.isfinite(val):
            return float(val)
        return -_np.inf
    except Exception:
        return -_np.inf

def _split_contiguous_folds(index, k):
    # K contiguous, non-overlapping folds (for robust scoring)
    idx = _np.arange(len(index))
    return _np.array_split(idx, k)

def optimize_weights_smart_sharpe(
    R,                      # DataFrame (Date x Combo) of simple daily returns, NaN already filled with 0
    k_folds=1,              # =1 -> no CV; >1 -> robustify by worst-fold SS (maximin)
    w_cap=0.35,             # per-combo upper bound
    n_starts=8,             # multi-start to avoid local optima
    random_state=42,
):
    """
    Solve: max_w SmartSharpe(R @ w)  s.t. sum(w)=1, 0<=w<=w_cap
    If k_folds>1, objective is maximin: maximize the minimum Smart Sharpe across K contiguous folds.
    """
    _rng = _np.random.default_rng(random_state)
    Rm = _np.asarray(R, dtype=float)
    T, N = Rm.shape
    if N == 0:
        raise ValueError("R has zero columns (no combos).")
    if T < 10:
        raise ValueError("Not enough time points to optimize.")

    # Constraints and bounds
    cons = ({'type': 'eq', 'fun': lambda w: _np.sum(w) - 1.0},)
    bnds = tuple((0.0, float(w_cap)) for _ in range(N))

    # Prepare folds
    folds = _split_contiguous_folds(R.index, k_folds) if k_folds and k_folds > 1 else [ _np.arange(T) ]

    def score_from_weights(w):
        # Return a **negative** value for minimizer (we maximize SS)
        # If k_folds>1: use the **min** SS across folds (maximin robustness)
        if (w < -1e-12).any() or (_np.sum(w) <= 0):
            return 1e6
        w = w / _np.sum(w)
        if k_folds and k_folds > 1:
            ss_vals = []
            for fold in folds:
                port = Rm[fold, :] @ w
                ss_vals.append(_smart_sharpe_safe(port))
            # maximize the worst fold
            val = _np.min(ss_vals)
        else:
            port = Rm @ w
            val = _smart_sharpe_safe(port)
        # Minimizer -> return negative
        return -float(val if _np.isfinite(val) else -1e9)

    # Candidates: equal, cap-spread, and random Dirichlet starts
    starts = []
    starts.append(_np.full(N, 1.0/N))  # equal
    if w_cap < 1.0:
        # "cap-spread" init: fill up to cap sequentially, then normalize
        k = int(_np.floor(1.0 / w_cap))
        s = _np.zeros(N)
        s[:min(N, k)] = w_cap
        rem = max(0.0, 1.0 - s.sum())
        if N > k and rem > 0:
            s[k] = rem
        s = s / s.sum()
        starts.append(s)
    for _ in range(max(0, n_starts - len(starts))):
        v = _rng.dirichlet(_np.ones(N))
        # squash any that exceed cap, then renormalize
        v = _np.minimum(v, w_cap)
        if v.sum() == 0:
            v = _np.full(N, 1.0/N)
        else:
            v = v / v.sum()
        starts.append(v)

    best = None
    for v0 in starts:
        try:
            res = _minimize(score_from_weights, v0, method='SLSQP', bounds=bnds, constraints=cons,
                            options={'maxiter': 1000, 'ftol': 1e-9})
            if not res.success:
                # keep even failed ones if they improve the objective
                pass
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    if best is None:
        raise RuntimeError("Optimization failed from all starts.")

    w = best.x
    w = _np.maximum(w, 0.0)
    w = w / w.sum()

    # Final metrics on full series
    port = _pd.Series(Rm @ w, index=R.index, name="portfolio_ssopt")
    port.reset_index(drop=True, inplace=True)
    ss = _smart_sharpe_safe(port)
    sh = _qs.stats.sharpe(port)
    so = _qs.stats.sortino(port)
    mdd = _qs.stats.max_drawdown(port)
    print(f"[SS-OPT] combos={N}  SmartSharpe={ss:.3f}  Sharpe={sh:.3f}  Sortino={so:.3f}  MaxDD={mdd:.2%}")

    # Also report fold SS if k_folds>1
    if k_folds and k_folds > 1:
        vals = []
        for i, fold in enumerate(folds, 1):
            vals.append(_smart_sharpe_safe(port.iloc[fold]))
        print(f"[SS-OPT] fold SS (min/median/max): {min(vals):.3f} / {_np.median(vals):.3f} / {max(vals):.3f}")

    return w, port

# ---- OPTIONAL: build a portfolio from the low-corr shortlist and compute Smart Sharpe ----
def build_portfolio_smart_sharpe(all_results, shortlist_csv_path, weight_scheme="equal"):
    import pandas as pd
    import numpy as np
    import quantstats as qs

    # 1) Read shortlist of combos (Signal,Ticker columns must exist)
    sh = pd.read_csv(shortlist_csv_path)
    if len(sh) == 0:
        print("[portfolio] shortlist empty; skipping.")
        return None, None, None

    # 2) Collect each combo's OOS daily return series from all_results
    #    We gather ONLY test/OOS series (the script stores them as 'Signal Returns' in the merged test results).
    def _collect_oos_series(df):
        # Expect columns: Signal, Ticker, Signal Returns (pd.Series), plus window tags
        if df is None or len(df) == 0:
            return []
        out = []
        for _, row in df.iterrows():
            if 'Signal Returns' in row and isinstance(row['Signal Returns'], pd.Series):
                out.append((row['Signal'], row['Ticker'], row['Signal Returns']))
        return out

    oos_containers = []
    h = all_results.get('holdout')
    if isinstance(h, dict) and len(h.get('filtered_results', [])) > 0:
        oos_containers.extend(_collect_oos_series(h['filtered_results']))

    for key in ('walk_forward','expanding','rolling'):
        df = all_results.get(key)
        if df is not None and len(df) > 0:
            oos_containers.extend(_collect_oos_series(df))

    # 3) Keep only shortlisted combos; combine windows by aligning on index and taking the value when non-NaN
    wanted = set((s, t) for s, t in zip(sh['Signal'], sh['Ticker']))
    series_map = {}  # (Signal,Ticker) -> concatenated daily series
    for sig, tkr, ser in oos_containers:
        key = (sig, tkr)
        if key not in wanted:
            continue
        ser = ser.astype(float).replace([np.inf, -np.inf], np.nan)
        if key not in series_map:
            series_map[key] = ser.copy()
        else:
            # If the same combo appears in multiple windows, prefer non-NaNs (later windows "fill in" gaps)
            series_map[key] = series_map[key].combine_first(ser)

    if not series_map:
        print("[portfolio] none of the shortlisted combos have OOS series; skipping.")
        return None, None, None

    # 4) Build Date x Combo matrix (simple returns), fill NaNs with 0 (flat when inactive)
    cols = []
    for (sig, tkr), ser in series_map.items():
        ser.name = f"{sig} @ {tkr}"
        cols.append(ser)
    R = pd.concat(cols, axis=1).sort_index().fillna(0.0)

    # 5) Choose weights
    n = R.shape[1]
    if weight_scheme == "equal":
        w = np.full(n, 1.0/n)
    else:
        # hook for custom weighting later (e.g., SS-optimized)
        w = np.full(n, 1.0/n)

    # 6) Portfolio daily series and Smart Sharpe
    port = pd.Series(R.values @ w, index=R.index, name="portfolio")
    port.reset_index(drop=True, inplace=True)
    smart_sharpe = qs.stats.smart_sharpe(port)
    sharpe = qs.stats.sharpe(port)
    sortino = qs.stats.sortino(port)

    print(f"[portfolio] combos={n}  SmartSharpe={smart_sharpe:.3f}  Sharpe={sharpe:.3f}  Sortino={sortino:.3f}")

    return port, R, w

# ---- /SMART SHARPE WEIGHT OPTIMIZATION ----

def fmt_pct(x):
    try:
        return f"{100 * float(x):.2f}%"
    except Exception:
        return "n/a"

def _drop_heavy_cols(df):
    """Drop heavy columns that bloat CSV files"""
    return df.drop(columns=['Signal Returns'], errors='ignore')

def save_method_averages(all_results, paths, name_prefix="", per_method_files=False):
    """
    For methods with multiple OOS test periods (walk-forward, expanding, rolling),
    compute per-signal averages & stability metrics across iterations and save to CSV.
    """
    import numpy as np
    import pandas as pd

    method_map = {
        'walk_forward':  'Walk-Forward',
        'expanding':     'Expanding Window',
        'rolling':       'Rolling Window',
    }

    frames = []
    for k, method_name in method_map.items():
        df = all_results.get(k, pd.DataFrame())
        if df is None or len(df) == 0:
            continue

        # keep only columns we can aggregate reliably
        keep = [c for c in df.columns if c in {
            'Signal','Ticker','Total Return','Smart Sharpe','Sharpe Ratio','Sortino Ratio',
            'Calmar Ratio','Max Drawdown','Time in Market','Robustness_Score'
        }]
        d = df[keep].copy()
        d['pos_ret_flag'] = (d['Total Return'] > 0).astype(float)
        d['pos_sharpe_flag'] = (d['Smart Sharpe'] > 0).astype(float)

        # group by Signal+Ticker to avoid mixing different targets
        g = d.groupby(['Signal','Ticker'], as_index=False)

        agg = g.agg({
            'Total Return':   ['mean','median','std','min','max','count'],
            'Smart Sharpe':   ['mean','median','std','min','max'],
            'Sharpe Ratio':   ['mean','median','std'],
            'Sortino Ratio':  ['mean','median','std'],
            'Calmar Ratio':   ['mean','median','std'],
            'Max Drawdown':   ['mean','median','std','min','max'],
            'Time in Market': ['mean','median'],
            'Robustness_Score':['mean','median'],
            'pos_ret_flag':   ['mean'],
            'pos_sharpe_flag':['mean'],
        })

        # flatten MultiIndex columns
        agg.columns = ['_'.join([c for c in col if c]).strip('_') for col in agg.columns.values]

        # add stability helpers (CoV = std / |mean|, guarded)
        def _cov(std_col, mean_col):
            m = agg[mean_col].replace(0, np.nan).abs()
            return (agg[std_col] / m).replace([np.inf, -np.inf], np.nan)

        agg['Sharpe_CoV'] = _cov('Smart Sharpe_std', 'Smart Sharpe_mean')
        agg['Return_CoV'] = _cov('Total Return_std', 'Total Return_mean')

        # --- Add robust distributional stats & ranks (PATCH) ---
        # Percentiles per group
        def _pct(s, q):
            try:
                return np.nanpercentile(s, q)
            except Exception:
                return np.nan

        # Geometric mean of (1+return) minus 1 (guard for negatives causing invalid product)
        def _gmean_return(x):
            x = pd.Series(x).dropna()
            # clamp extreme negatives at -0.999 to avoid invalid product
            x = x.clip(lower=-0.999)
            return float(np.exp(np.log1p(x)).mean() - 1) if len(x) else np.nan

        # Recompute on the raw grouped object so we can access series
        extras = g.apply(lambda df: pd.Series({
            'Sharpe_p10': _pct(df['Smart Sharpe'], 10),
            'Sharpe_p25': _pct(df['Smart Sharpe'], 25),
            'Sharpe_p50': _pct(df['Smart Sharpe'], 50),   # median
            'Sharpe_p75': _pct(df['Smart Sharpe'], 75),
            'Return_p10': _pct(df['Total Return'], 10),   # worst decile
            'Return_p50': _pct(df['Total Return'], 50),   # median
            'Return_gmean': _gmean_return(df['Total Return']),
            'MaxDD_p90': _pct(df['Max Drawdown'], 90),    # tail drawdown (more negative is worse)
        })).reset_index()

        # Merge extras back
        agg = agg.merge(extras, on=['Signal','Ticker'], how='left')

        # Stability coefficients (already added CoV; add IQR for Sharpe)
        agg['Sharpe_IQR'] = agg['Sharpe_p75'] - agg['Sharpe_p25']

        # Suggested robust ranking key (higher is better):
        #  - primary: median Smart Sharpe
        #  - tie-breaks: geometric return, hit-rate on Sharpe, lower dispersion, and tail DD
        agg['RankKey'] = (
            agg['Sharpe_p50']
            .fillna(-np.inf)
        ).astype(float)

        # hit rates
        agg.rename(columns={
            'pos_ret_flag_mean': 'HitRate_Positive_Return',
            'pos_sharpe_flag_mean': 'HitRate_Positive_Sharpe',
            'Total Return_count': 'N_Iterations'
        }, inplace=True)

        # cosmetic & method column
        agg.insert(0, 'Method', method_name)

        if per_method_files:
            # Sort by robust ranking key (descending) with tie-breakers
            agg_sorted = agg.sort_values(
                ['RankKey', 'Return_gmean', 'HitRate_Positive_Sharpe', 'Sharpe_IQR', 'MaxDD_p90'],
                ascending=[False, False, False, True, False]
            )
            folder = {
                'walk_forward': paths.walk_forward,
                'expanding': paths.expanding,
                'rolling': paths.rolling
            }[k]
            save_df(agg_sorted, folder, "averages.csv", kind="averages", method=k)

            # Save combo-only averages file
            if len(agg_sorted) > 0:
                combo_df = agg_sorted[has_plus(agg_sorted['Signal'])]
                if len(combo_df) > 0:
                    combo_path = os.path.join(folder, f"{name_prefix}_{k}_averages_combos_only.csv")
                    _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)

        frames.append(agg)

    if frames:
        all_avg = pd.concat(frames, ignore_index=True)
        # Sort combined file by robust ranking key (descending) with tie-breakers
        all_avg_sorted = all_avg.sort_values(
            ['RankKey', 'Return_gmean', 'HitRate_Positive_Sharpe', 'Sharpe_IQR', 'MaxDD_p90'],
            ascending=[False, False, False, True, False]
        )
        save_df(all_avg_sorted, paths.aggregates, "method_averages.csv",
                kind="combined_averages", method="all")

        # Save combined combo-only averages file
        if len(all_avg_sorted) > 0:
            combo_df = all_avg_sorted[has_plus(all_avg_sorted['Signal'])]
            if len(combo_df) > 0:
                combo_path = os.path.join(paths.aggregates, f"{name_prefix}_all_methods_averages_combos_only.csv")
                _drop_heavy_cols(combo_df).to_csv(combo_path, index=False)
    else:
        print("No multi-period method results to average.")

# === FINAL, SILENT, CRASH-PROOF VERSION of generate_evaluation_summary ===
def generate_evaluation_summary(all_results, paths, name_prefix="", top_n_per_method=50):
    """
    Generate a comprehensive summary report CSV.
    Console output is silenced. Includes robust checks for missing data keys.
    """
    print(f"\n{'='*60}")
    print("GENERATING EVALUATION SUMMARY (CSV ONLY)")
    print(f"{'='*60}")

    summary_data = []

    # Process holdout results
    if isinstance(all_results, dict) and 'holdout' in all_results:
        holdout = all_results['holdout']
        # === FIX: Safe access using .get() ===
        filtered_res = holdout.get('filtered_results')
        
        if isinstance(filtered_res, pd.DataFrame) and not filtered_res.empty:
            top_rows = filtered_res.sort_values('Smart Sharpe', ascending=False).head(top_n_per_method)
            for rank, (_, row) in enumerate(top_rows.iterrows(), start=1):
                summary_data.append({
                    'Method': 'Holdout (70/30)',
                    'Rank': rank,
                    'Signal': str(row['Signal']),
                    'Ticker': row['Ticker'],
                    'OOS_Return': fmt_pct(row['Total Return']),
                    'OOS_Sharpe': f"{row['Smart Sharpe']:.3f}",
                    'OOS_Max_DD': fmt_pct(row['Max Drawdown']),
                    'Robustness': f"{row['Robustness_Score']:.3f}",
                    'Train_Period': holdout.get('train_period', 'N/A'),
                    'Test_Period': holdout.get('test_period', 'N/A'),
                    'Total_Signals': len(filtered_res),
                    'Notes': f"Embargo: {holdout.get('embargo_days', 0)} days"
                })

    # Process walk-forward results
    if 'walk_forward' in all_results and isinstance(all_results['walk_forward'], pd.DataFrame) and not all_results['walk_forward'].empty:
        wf_data = all_results['walk_forward']
        required_cols = ['Signal', 'Ticker', 'Total Return', 'Smart Sharpe', 'Max Drawdown', 'Robustness_Score']
        if all(col in wf_data.columns for col in required_cols):
            avg = (wf_data.groupby(['Signal','Ticker'], as_index=False)
                   .agg({'Total Return':['median'],
                         'Smart Sharpe':['median'],
                         'Max Drawdown':['median'],
                         'Robustness_Score':['median']}))
            avg.columns = ['Signal','Ticker','Total Return','Smart Sharpe','Max Drawdown','Robustness_Score']
            avg = avg.sort_values(['Smart Sharpe','Total Return'], ascending=[False, False])
            top_rows = avg.head(top_n_per_method)
            for rank, row in enumerate(top_rows.iterrows(), start=1):
                r = row[1] 
                summary_data.append({
                    'Method': 'Walk-Forward',
                    'Rank': rank,
                    'Signal': str(r['Signal']),
                    'Ticker': r['Ticker'],
                    'OOS_Return': fmt_pct(r['Total Return']),
                    'OOS_Sharpe': f"{r['Smart Sharpe']:.3f}",
                    'OOS_Max_DD': fmt_pct(r['Max Drawdown']),
                    'Robustness': f"{r['Robustness_Score']:.3f}",
                    'Train_Period': 'Multiple',
                    'Test_Period': 'Multiple',
                    'Total_Signals': len(avg),
                    'Notes': f"Avg across {wf_data['WF_Iteration'].nunique() if 'WF_Iteration' in wf_data.columns else '?'} iterations"
                })

    # Process expanding window results
    if 'expanding' in all_results and isinstance(all_results['expanding'], pd.DataFrame) and not all_results['expanding'].empty:
        ew_data = all_results['expanding']
        required_cols = ['Signal', 'Ticker', 'Total Return', 'Smart Sharpe', 'Max Drawdown', 'Robustness_Score']
        if all(col in ew_data.columns for col in required_cols):
            avg = (ew_data.groupby(['Signal','Ticker'], as_index=False)
                   .agg({'Total Return':['median'],
                         'Smart Sharpe':['median'],
                         'Max Drawdown':['median'],
                         'Robustness_Score':['median']}))
            avg.columns = ['Signal','Ticker','Total Return','Smart Sharpe','Max Drawdown','Robustness_Score']
            avg = avg.sort_values(['Smart Sharpe','Total Return'], ascending=[False, False])
            top_rows = avg.head(top_n_per_method)
            for rank, row in enumerate(top_rows.iterrows(), start=1):
                r = row[1]
                summary_data.append({
                    'Method': 'Expanding Window',
                    'Rank': rank,
                    'Signal': str(r['Signal']),
                    'Ticker': r['Ticker'],
                    'OOS_Return': fmt_pct(r['Total Return']),
                    'OOS_Sharpe': f"{r['Smart Sharpe']:.3f}",
                    'OOS_Max_DD': fmt_pct(r['Max Drawdown']),
                    'Robustness': f"{r['Robustness_Score']:.3f}",
                    'Train_Period': 'Expanding',
                    'Test_Period': 'Multiple',
                    'Total_Signals': len(avg),
                    'Notes': f"Avg across {ew_data['EW_Iteration'].nunique() if 'EW_Iteration' in ew_data.columns else '?'} iterations"
                })

    # Process rolling window results
    if 'rolling' in all_results and isinstance(all_results['rolling'], pd.DataFrame) and not all_results['rolling'].empty:
        roll_data = all_results['rolling']
        required_cols = ['Signal', 'Ticker', 'Total Return', 'Smart Sharpe', 'Max Drawdown', 'Robustness_Score']
        if all(col in roll_data.columns for col in required_cols):
            avg = (roll_data.groupby(['Signal','Ticker'], as_index=False)
                   .agg({'Total Return':['median'],
                         'Smart Sharpe':['median'],
                         'Max Drawdown':['median'],
                         'Robustness_Score':['median']}))
            avg.columns = ['Signal','Ticker','Total Return','Smart Sharpe','Max Drawdown','Robustness_Score']
            avg = avg.sort_values(['Smart Sharpe','Total Return'], ascending=[False, False])
            top_rows = avg.head(top_n_per_method)
            for rank, row in enumerate(top_rows.iterrows(), start=1):
                r = row[1]
                summary_data.append({
                    'Method': 'Rolling Window',
                    'Rank': rank,
                    'Signal': str(r['Signal']),
                    'Ticker': r['Ticker'],
                    'OOS_Return': fmt_pct(r['Total Return']),
                    'OOS_Sharpe': f"{r['Smart Sharpe']:.3f}",
                    'OOS_Max_DD': fmt_pct(r['Max Drawdown']),
                    'Robustness': f"{r['Robustness_Score']:.3f}",
                    'Train_Period': 'Fixed Window',
                    'Test_Period': 'Multiple',
                    'Total_Signals': len(avg),
                    'Notes': f"Avg across {roll_data['Roll_Iteration'].nunique() if 'Roll_Iteration' in roll_data.columns else '?'} iterations"
                })

    # Create summary DataFrame and save
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary
        save_df(summary_df, paths.aggregates, "evaluation_summary.csv",
                kind="summary", method="all")

        # Save combo-only summary file
        if len(summary_df) > 0:
            combo_df = summary_df[has_plus(summary_df['Signal'])]
            if len(combo_df) > 0:
                combo_path = os.path.join(paths.aggregates, f"{name_prefix}_evaluation_summary_combos_only.csv")
                combo_df.to_csv(combo_path, index=False)

        # Recommendations / risk
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")

        holdout_dd = 0.0
        # === FIX: Robust check for holdout data ===
        if isinstance(all_results, dict) and 'holdout' in all_results:
            holdout_res = all_results['holdout'].get('filtered_results')
            if isinstance(holdout_res, pd.DataFrame) and not holdout_res.empty:
                holdout_dd = abs(holdout_res.iloc[0]['Max Drawdown'])

        if holdout_dd > 0.15:
            print(f"⚠️  HIGH RISK: Primary method shows {fmt_pct(holdout_dd)} max drawdown")
            print("   Consider position sizing adjustments or additional risk management")
        elif holdout_dd > 0.08:
            print(f"⚠️  MODERATE RISK: Primary method shows {fmt_pct(holdout_dd)} max drawdown")
            print("   Standard position sizing appropriate")
        else:
            print(f"✅ LOW RISK: Primary method shows {fmt_pct(holdout_dd)} max drawdown")
            print("   Conservative position sizing may be overly cautious")

    print(f"\nEvaluation complete! All results saved to: {paths.root}/")

def get_enhanced_user_inputs():
    """Enhanced user input function with evaluation method selection"""
    print("=== Enhanced Signal Backtesting with Multiple Evaluation Methods ===")

    # === ADD THIS NEW BLOCK AT THE TOP ===
    print("\nPlease define a name for this experimental run.")
    print("If you use a name that already exists, the script will attempt to resume that run.")
    default_name = f"run_{datetime.now().strftime('%Y%m%d')}"
    run_name = safe_input("Enter Run Name", default=default_name)
    config = {'run_name': run_name}  # Start the config dict with the run name
    # =====================================

    print("\nSystem Resources:")
    max_workers_input = safe_input(
        "Max parallel workers (lower = less RAM; default 5, recommended for 16 GB+ systems): ",
        default="5"
    ).strip()
    config['max_workers'] = int(max_workers_input) if max_workers_input.isdigit() else 5

    print("Select evaluation methods to run:\n")

    # Get basic configuration first
    config.update(get_user_inputs())  # Original function

    # Add evaluation method selection
    print("\nEvaluation Methods:")
    print("1. Holdout Split (70/30 with embargo) - Fast, good for initial screening")
    print("2. Walk-Forward Analysis - Rolling windows, good for stability assessment")
    print("3. Expanding Window - Growing training set, good for regime analysis")
    print("4. All Methods - Comprehensive evaluation (recommended)")
    print("5. Rolling Window - Fixed training window over time")

    raw = safe_input("Select evaluation methods (comma-separated, e.g., 2,5) [4]: ", default="4").strip()
    choices = {c.strip() for c in raw.split(',') if c.strip()} or {"4"}  # default to 'All' if blank

    eval_modes = []
    if "4" in choices:
        eval_modes = [EvalMode.HOLDOUT_70_30, EvalMode.WALK_FORWARD, EvalMode.EXPANDING, EvalMode.ROLLING]
    else:
        if "1" in choices: eval_modes.append(EvalMode.HOLDOUT_70_30)
        if "2" in choices: eval_modes.append(EvalMode.WALK_FORWARD)
        if "3" in choices: eval_modes.append(EvalMode.EXPANDING)
        if "5" in choices: eval_modes.append(EvalMode.ROLLING)

    # Get evaluation configuration
    eval_config = EvaluationConfig()

    if EvalMode.HOLDOUT_70_30 in eval_modes:
        train_pct = safe_input(f"Holdout training percentage [70]: ", default="70").strip()
        if train_pct:
            eval_config.holdout_train_pct = float(train_pct) / 100.0

        embargo_days = safe_input(f"Embargo days between train/test [5]: ", default="5").strip()
        if embargo_days:
            eval_config.embargo_days = int(embargo_days)

    if EvalMode.WALK_FORWARD in eval_modes:
        wf_train = safe_input(f"Walk-forward training window days [252]: ", default="252").strip()
        if wf_train:
            eval_config.wf_train_period = int(wf_train)

        wf_test = safe_input(f"Walk-forward test window days [63]: ", default="63").strip()
        if wf_test:
            eval_config.wf_test_period = int(wf_test)

        wf_step = safe_input(f"Walk-forward step size days [21]: ", default="21").strip()
        if wf_step:
            eval_config.wf_step_size = int(wf_step)

    if EvalMode.EXPANDING in eval_modes:
        exp_initial = safe_input(f"Expanding window initial training days [252]: ", default="252").strip()
        if exp_initial:
            eval_config.exp_initial_train = int(exp_initial)

        exp_test = safe_input(f"Expanding window test period days [63]: ", default="63").strip()
        if exp_test:
            eval_config.exp_test_period = int(exp_test)

        exp_expansion = safe_input(f"Expanding window expansion size days [63]: ", default="63").strip()
        if exp_expansion:
            eval_config.exp_expansion_size = int(exp_expansion)

    if EvalMode.ROLLING in eval_modes:
        roll_train = safe_input(f"Rolling training window days [252]: ", default="252").strip()
        if roll_train:
            eval_config.roll_train_period = int(roll_train)
        roll_test = safe_input(f"Rolling test period days [63]: ", default="63").strip()
        if roll_test:
            eval_config.roll_test_period = int(roll_test)
        roll_step = safe_input(f"Rolling step size days [21]: ", default="21").strip()
        if roll_step:
            eval_config.roll_step_size = int(roll_step)

    # Add evaluation config to main config
    config['eval_modes'] = eval_modes
    config['eval_config'] = eval_config

    # Add synergistic combo configuration
    print("\nSynergistic Signal Combination Settings:")
    print("This will generate AND/OR/gated combinations of signals to find synergistic pairs.")

    enable_combos = safe_input("Enable synergistic signal combinations? [Y/n]: ", default="Y").strip().lower()
    if enable_combos != 'n':
        config['enable_synergistic_combos'] = True

        k_primary = safe_input("Top K primary signals to consider [30]: ", default="30").strip()
        config['k_primary'] = int(k_primary) if k_primary else 30

        m_partner = safe_input("Number of partner signals to sample [40]: ", default="40").strip()
        config['m_partner'] = int(m_partner) if m_partner else 40

        min_train_gain = safe_input("Minimum train improvement over best member [0.05]: ", default="0.05").strip()
        config['min_train_gain'] = float(min_train_gain) if min_train_gain else 0.05

        min_test_gain = safe_input("Minimum test improvement over best member [0.00]: ", default="0.00").strip()
        config['min_test_gain'] = float(min_test_gain) if min_test_gain else 0.00

        max_legs = safe_input("Max combo legs [2]: ", default="2").strip()
        config['max_combo_legs'] = int(max_legs) if max_legs else 2

        print("✓ Synergistic combinations enabled")
    else:
        config['enable_synergistic_combos'] = False
        config['k_primary'] = 30
        config['m_partner'] = 40
        config['min_train_gain'] = 0.05
        config['min_test_gain'] = 0.00
        config['max_combo_legs'] = 2
        print("✗ Synergistic combinations disabled")

    # Add frozen combo universe configuration
    if COMBO_MODULES_AVAILABLE:
        print("\nFrozen Combo Universe Settings:")
        print("This creates a stable set of combos evaluated across all windows for consistency.")

        enable_frozen = safe_input("Enable frozen combo universe? [Y/n]: ", default="Y").strip().lower()
        if enable_frozen != 'n':
            config['freeze_combo_universe'] = True

            universe_size = safe_input("Number of combos to freeze [50]: ", default="50").strip()
            config['combo_universe_size'] = int(universe_size) if universe_size else 50

            universe_source = safe_input("Freeze from holdout train or first WF train? [holdout_train/first_wf_train]: ", default="holdout_train").strip()
            config['combo_universe_source'] = universe_source if universe_source else "holdout_train"

            print("✓ Frozen combo universe enabled")
        else:
            config['freeze_combo_universe'] = False
            config['combo_universe_size'] = 50
            config['combo_universe_source'] = 'holdout_train'
            print("✗ Frozen combo universe disabled")

        # Optional correlation and portfolio settings
        print("\nOptional Advanced Settings:")
        enable_advanced = safe_input("Enable correlation analysis and portfolio construction? [Y/n]: ", default="Y").strip().lower()
        if enable_advanced != 'n':
            min_overlap = safe_input("Minimum date overlap for correlations [60]: ", default="60").strip()
            config['combo_corr_min_overlap'] = int(min_overlap) if min_overlap else 60

            portfolio_method = safe_input("Portfolio method (invvol/erc) [invvol]: ", default="invvol").strip()
            config['portfolio_method'] = portfolio_method if portfolio_method else "invvol"

            # Smart-Sharpe optimizer settings
            smart_sharpe = PM.ask_bool_once(
                key="smart_sharpe",
                question="Enable Smart-Sharpe portfolio optimization?",
                default=False,
            )
            config['ssopt_enable'] = smart_sharpe
            config['enable_ssopt'] = smart_sharpe  # Keep both for compatibility
            if smart_sharpe:
                config['ssopt_cfg'] = {
                    "w_floor": 0.0,
                    "w_cap": 0.35,
                    "en_min": 6,
                    "lambda_std": 0.25,
                    "corr_penalty": 0.0,
                    "n_boot": 200
                }
                print("✓ Smart-Sharpe optimization enabled")
            else:
                config['ssopt_cfg'] = {}
                print("✓ Smart-Sharpe optimization disabled")
        else:
            config['combo_corr_min_overlap'] = 60
            config['portfolio_method'] = 'invvol'
            config['ssopt_enable'] = False
            config['enable_ssopt'] = False  # Keep both for compatibility
            config['ssopt_cfg'] = {}
            print("✗ Advanced features disabled")
    else:
        # Set defaults if modules not available
        config['freeze_combo_universe'] = False
        config['combo_universe_size'] = 50
        config['combo_universe_source'] = 'holdout_train'
        config['combo_corr_min_overlap'] = 60
        config['portfolio_method'] = 'invvol'
        config['ssopt_enable'] = False
        config['enable_ssopt'] = False  # Keep both for compatibility
        config['ssopt_cfg'] = {}
        print("⚠️ Combo modules not available - advanced features disabled")

    # Add optional combo export configuration
    print("\nCombo Export Settings (Optional):")
    print("These control the frozen combo universe and portfolio construction.")

    frozen_top_k = safe_input("Number of combos to freeze across windows [30]: ", default="30").strip()
    config['frozen_top_k'] = int(frozen_top_k) if frozen_top_k else 30

    corr_threshold = safe_input("Correlation threshold for low-correlation shortlist [0.30]: ", default="0.30").strip()
    config['combo_corr_threshold'] = float(corr_threshold) if corr_threshold else 0.30

    shortlist_size = safe_input("Maximum size of low-correlation shortlist [12]: ", default="12").strip()
    config['combo_shortlist_size'] = int(shortlist_size) if shortlist_size else 12

    # Portfolio optimization settings (using single Smart-Sharpe flag from above)
    print("\nPortfolio Optimization Settings:")
    smart_sharpe = PM.get("smart_sharpe", False)
    print(f"(Smart-Sharpe already set: {'ON' if smart_sharpe else 'OFF'})")

    if smart_sharpe:
        k_folds = safe_input("Cross-validation folds for robust optimization [3]: ", default="3").strip()
        config['ssopt_k_folds'] = int(k_folds) if k_folds else 3

        w_cap = safe_input("Maximum weight per combo [0.35]: ", default="0.35").strip()
        config['ssopt_w_cap'] = float(w_cap) if w_cap else 0.35

        n_starts = safe_input("Number of optimization starts [12]: ", default="12").strip()
        config['ssopt_n_starts'] = int(n_starts) if n_starts else 12

        print(f"✓ Portfolio optimization configured: {config['ssopt_k_folds']} folds, weight cap {config['ssopt_w_cap']}, {config['ssopt_n_starts']} starts")
    else:
        print("✓ Portfolio optimization disabled (Smart-Sharpe not enabled above)")

    print(f"✓ Combo export configured: freeze top {config['frozen_top_k']}, corr threshold {config['combo_corr_threshold']}, shortlist size {config['combo_shortlist_size']}")

    # Add preconditions configuration
    print("\nSignal Preconditions (Optional):")
    print("These filter signals based on market conditions before backtesting.")
    preconds, precond_combine = get_preconditions_from_user()
    config['preconditions'] = preconds
    config['precondition_combine'] = precond_combine

    if preconds:
        print(f"✓ Preconditions configured: {len(preconds)} expression(s) with {precond_combine} logic")
    else:
        print("✓ No preconditions - all signals will be evaluated")

    return config

def get_eval_only():
    """Ask only for evaluation modes + their params (reuse tickers/signals)."""
    _cleanup_tqdm()  # <--- add this
    print("\nEvaluation Methods:")
    print("1. Holdout (70/30 + embargo)")
    print("2. Walk-Forward")
    print("3. Expanding Window")
    print("4. All Methods")
    print("5. Rolling Window")
    raw = safe_input("Select evaluation methods (comma-separated, e.g., 2,5) [4]: ", default="4").strip()
    choices = {c.strip() for c in raw.split(',') if c.strip()} or {"4"}  # default to 'All' if blank

    modes = []
    if "4" in choices:
        modes = [EvalMode.HOLDOUT_70_30, EvalMode.WALK_FORWARD, EvalMode.EXPANDING, EvalMode.ROLLING]
    else:
        if "1" in choices: modes.append(EvalMode.HOLDOUT_70_30)
        if "2" in choices: modes.append(EvalMode.WALK_FORWARD)
        if "3" in choices: modes.append(EvalMode.EXPANDING)
        if "5" in choices: modes.append(EvalMode.ROLLING)

    cfg = EvaluationConfig()
    if EvalMode.HOLDOUT_70_30 in modes:
        x = safe_input("Holdout training % [70]: ", default="70").strip()
        if x: cfg.holdout_train_pct = float(x)/100.0
        x = safe_input("Embargo days [5]: ", default="5").strip()
        if x: cfg.embargo_days = int(x)
    if EvalMode.WALK_FORWARD in modes:
        x = safe_input("WF train days [252]: ", default="252").strip()
        if x: cfg.wf_train_period = int(x)
        x = safe_input("WF test  days [63]: ", default="63").strip()
        if x: cfg.wf_test_period = int(x)
        x = safe_input("WF step  days [21]: ", default="21").strip()
        if x: cfg.wf_step_size = int(x)
    if EvalMode.EXPANDING in modes:
        x = safe_input("Exp init train days [252]: ", default="252").strip()
        if x: cfg.exp_initial_train = int(x)
        x = safe_input("Exp test days [63]: ", default="63").strip()
        if x: cfg.exp_test_period = int(x)
        x = safe_input("Exp expansion size [63]: ", default="63").strip()
        if x: cfg.exp_expansion_size = int(x)
    if EvalMode.ROLLING in modes:
        x = safe_input("Roll train days [252]: ", default="252").strip()
        if x: cfg.roll_train_period = int(x)
        x = safe_input("Roll test days [63]: ", default="63").strip()
        if x: cfg.roll_test_period = int(x)
        x = safe_input("Roll step days [21]: ", default="21").strip()
        if x: cfg.roll_step_size = int(x)

    return modes, cfg

# === ADD THE MISSING generate_signals FUNCTION BACK IN HERE ===
def generate_signals(
    tickers,
    price_data,
    signal_types,
    rsi_periods=None,
    price_sma_periods=None,
    price_ema_periods=None,
    returns_ma_periods=None
):
    """Generate trading signals based on selected types using price_data"""
    daily_returns = price_data.pct_change()
    log_returns = np.log(price_data / price_data.shift(1))

    signals = {}

    safe_print(f"Generating {signal_types} signals...")

    if 'RSI' in signal_types:
        print("Computing RSI indicators...")

        if rsi_periods is not None:
            unique_periods = set()
            for p1, p2 in rsi_periods:
                unique_periods.add(p1)
                unique_periods.add(p2)
            periods_to_compute = sorted(unique_periods)
            print(f"Custom RSI periods: {periods_to_compute}")
        else:
            periods_to_compute = list(range(5, 35, 5))
            print(f"Default RSI periods: {periods_to_compute}")

        rsi_cache = {}
        for t in tqdm(tickers, desc="Computing RSI indicators", leave=False):
            rsi_cache[t] = {}
            for p in periods_to_compute:
                rsi_cache[t][p] = RSIIndicator(close=_series(price_data, t), window=p).rsi()

    if 'CUMRET' in signal_types:
        print("Computing cumulative return indicators...")
        cumret_cache = {}
        for t in tqdm(tickers, desc="Computing cumulative return indicators", leave=False):
            cumret_cache[t] = {}
            for p in range(5, 95, 5):
                cumret_cache[t][p] = (np.exp(_series(log_returns, t).rolling(p).sum()) - 1)

    if 'RETURNS_MA' in signal_types:
        print("Computing moving average of returns indicators...")
        returns_ma_cache = {}
        periods = returns_ma_periods if returns_ma_periods else list(range(10, 110, 10))
        for t in tqdm(tickers, desc="Computing returns moving average indicators", leave=False):
            returns_ma_cache[t] = {}
            for p in periods:
                returns_ma_cache[t][p] = _series(daily_returns, t).rolling(p).mean()

    if 'STD' in signal_types:
        print("Computing standard deviation indicators...")
        std_cache = {}
        for t in tqdm(tickers, desc="Computing standard deviation indicators", leave=False):
            std_cache[t] = {}
            for p in range(10, 60, 10):
                std_cache[t][p] = _series(daily_returns, t).rolling(p).std()

    if 'RSI' in signal_types:
        print("Generating RSI signals...")
        rsi_levels = range(10, 100, 10)
        for t in tickers:
            for p, rsi in rsi_cache[t].items():
                for lvl in rsi_levels:
                    signals[f'RSI_{p}_{t}_GT_{lvl}'] = rsi > lvl
                    signals[f'RSI_{p}_{t}_LT_{lvl}'] = rsi < lvl

        if rsi_periods is not None:
            print(f"Generating custom RSI comparisons: {rsi_periods}")
            for p1, p2 in rsi_periods:
                rsi_keys_p1 = [(t, p1) for t in tickers if p1 in rsi_cache.get(t, {})]
                rsi_keys_p2 = [(t, p2) for t in tickers if p2 in rsi_cache.get(t, {})]
                seen = set()
                for t1, _ in rsi_keys_p1:
                    for t2, _ in rsi_keys_p2:
                        a = (t1, p1); b = (t2, p2)
                        if a == b: continue
                        pair = tuple(sorted([a, b]))
                        if pair in seen: continue
                        seen.add(pair)
                        (t_lo, p_lo), (t_hi, p_hi) = pair
                        rsi_lo = rsi_cache[t_lo][p_lo]
                        rsi_hi = rsi_cache[t_hi][p_hi]
                        signals[f'RSI_{p_lo}_{t_lo}_GT_RSI_{p_hi}_{t_hi}'] = rsi_lo > rsi_hi
        else:
            rsi_keys = [(t, p) for t in tickers for p in rsi_cache.get(t, {}).keys()]
            for (t1, p1), (t2, p2) in itertools.combinations(rsi_keys, 2):
                rsi1 = rsi_cache[t1][p1]
                rsi2 = rsi_cache[t2][p2]
                signals[f'RSI_{p1}_{t1}_GT_RSI_{p2}_{t2}'] = rsi1 > rsi2

    if 'CUMRET' in signal_types:
        print("Generating cumulative return signals...")
        cumret_levels = [i / 100 for i in range(-10, 11, 2)]
        for t in tickers:
            for p, cum in cumret_cache[t].items():
                for lvl in cumret_levels:
                    signals[f'CUMRET_{p}_{t}_GT_{lvl}'] = cum > lvl
                    signals[f'CUMRET_{p}_{t}_LT_{lvl}'] = cum < lvl
        cumret_keys = [(t, p) for t in tickers for p in cumret_cache[t].keys()]
        for (t1, p1), (t2, p2) in itertools.combinations(cumret_keys, 2):
            r1 = cumret_cache[t1][p1]
            r2 = cumret_cache[t2][p2]
            signals[f'CUMRET_{p1}_{t1}_GT_CUMRET_{p2}_{t2}'] = r1 > r2

    if 'RETURNS_MA' in signal_types:
        print("Generating moving average of returns signals...")
        returns_ma_keys = [(t, p) for t in tickers for p in returns_ma_cache[t].keys()]
        for (t1, p1), (t2, p2) in itertools.combinations(returns_ma_keys, 2):
            m1 = returns_ma_cache[t1][p1]
            m2 = returns_ma_cache[t2][p2]
            signals[f'RETURNS_MA_{p1}_{t1}_GT_RETURNS_MA_{p2}_{t2}'] = m1 > m2

    if 'STD' in signal_types:
        print("Generating standard deviation signals...")
        std_keys = [(t, p) for t in tickers for p in std_cache[t].keys()]
        for (t1, p1), (t2, p2) in itertools.combinations(std_keys, 2):
            s1 = std_cache[t1][p1]
            s2 = std_cache[t2][p2]
            signals[f'STD_{p1}_{t1}_GT_STD_{p2}_{t2}'] = s1 > s2

    if 'PRICE_SMA' in signal_types:
        print("Generating PRICE SMA signals...")
        PRICE_SMA_PERIODS = price_sma_periods if price_sma_periods else [20, 30, 50, 100, 150, 200]
        price_sma_cache = {t: {p: price_data[t].rolling(p).mean() for p in PRICE_SMA_PERIODS} for t in tickers}
        print("Generating PRICE vs SMA(price) signals...")
        for t in tickers:
            px = _series(price_data, t)
            for p, sma in price_sma_cache[t].items():
                signals[f'PRICE_{t}_GT_SMA_{p}_{t}'] = px > sma
                signals[f'PRICE_{t}_LT_SMA_{p}_{t}'] = px < sma
        print("Generating SMA(price) vs SMA(price) signals...")
        sma_keys = [(t, p) for t in tickers for p in PRICE_SMA_PERIODS]
        for (t1, p1), (t2, p2) in itertools.combinations(sma_keys, 2):
            sma1 = price_sma_cache[t1][p1]
            sma2 = price_sma_cache[t2][p2]
            signals[f'SMA_{p1}_{t1}_GT_SMA_{p2}_{t2}'] = sma1 > sma2

    if 'PRICE_EMA' in signal_types:
        print("Generating PRICE EMA signals...")
        PRICE_EMA_PERIODS = price_ema_periods if price_ema_periods else [20, 30, 50, 100, 150, 200]
        price_ema_cache = {t: {p: price_data[t].ewm(span=p, adjust=False).mean() for p in PRICE_EMA_PERIODS} for t in tickers}
        print("Generating PRICE vs EMA(price) signals...")
        for t in tickers:
            px = _series(price_data, t)
            for p, ema in price_ema_cache[t].items():
                signals[f'PRICE_{t}_GT_EMA_{p}_{t}'] = px > ema
                signals[f'PRICE_{t}_LT_EMA_{p}_{t}'] = px < ema
        print("Generating EMA(price) vs EMA(price) signals...")
        ema_keys = [(t, p) for t in tickers for p in PRICE_EMA_PERIODS]
        for (t1, p1), (t2, p2) in itertools.combinations(ema_keys, 2):
            ema1 = price_ema_cache[t1][p1]
            ema2 = price_ema_cache[t2][p2]
            signals[f'EMA_{p1}_{t1}_GT_EMA_{p2}_{t2}'] = ema1 > ema2

    for k in signals:
        if not isinstance(signals[k], pd.Series):
            signals[k] = pd.Series(signals[k], index=price_data.index)

    print(f"Generated {len(signals)} signals")
    return signals
# ==========================================================

# === FINAL, CORRECTED VERSION of enhanced_main with robust resume logic ===
def enhanced_main():
    """
    Enhanced main execution function with a robust checkpoint/resume system.
    """
    try:
        # --- Get the user-defined Run Name first ---
        print("=== Enhanced Signal Backtesting with Multiple Evaluation Methods ===")
        print("\nPlease define a name for this experimental run.")
        print("If you use a name that already exists, the script will attempt to resume that run.")
        default_name = f"run_{datetime.now().strftime('%Y%m%d')}"
        run_name = safe_input("Enter Run Name", default=default_name)
        
        parent_dir = BASE_OUTPUT_DIR / run_name
        os.makedirs(parent_dir, exist_ok=True)
        master_checkpoint_path = parent_dir / "master_checkpoint.pkl"

        # --- Attempt to Resume from Master Checkpoint ---
        if os.path.exists(master_checkpoint_path):
            print("\n" + "="*20 + " RESUME " + "="*20)
            print(f"Master checkpoint found for run '{run_name}'. Attempting to resume.")
            try:
                with open(master_checkpoint_path, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                base_cfg = checkpoint_data['base_cfg']
                signals = checkpoint_data['signals']
                full_price_data = checkpoint_data['full_price_data']
                blackout_ranges = checkpoint_data['blackout_ranges']
                run_idx = checkpoint_data['run_idx']
                completed_evaluations = checkpoint_data.get('completed_evaluations', [])
                
                print("✓ Successfully loaded all data and configuration from checkpoint.")
            except Exception as e:
                print(f"  ✗ CRITICAL ERROR: Could not load master checkpoint file: {e}")
                return
        else:
            # --- If No Checkpoint, Start a Fresh Run ---
            print("\n" + "="*20 + " FRESH RUN " + "="*20)
            base_cfg = get_enhanced_user_inputs()
            base_cfg['run_name'] = run_name
            global MAX_WORKERS
            MAX_WORKERS = base_cfg.get('max_workers', 5)
            run_idx = 1
            completed_evaluations = []

            # (Data loading/caching/generation logic)
            cache_dir = Path("cache")
            cache_dir.mkdir(exist_ok=True)
            all_tickers_list = sorted(list(set(base_cfg.get('tickers', []) + base_cfg.get('target', []) + [base_cfg.get('benchmark', 'SPY')])))
            tickers_hash = str(hash(tuple(all_tickers_list)))
            prices_cache_file = cache_dir / f"prices_{tickers_hash}.pkl"
            signals_cache_file = cache_dir / f"signals_{tickers_hash}.pkl"
            
            signals, full_price_data = None, None
            if prices_cache_file.exists() and signals_cache_file.exists():
                try:
                    with open(prices_cache_file, "rb") as f: full_price_data = pickle.load(f)
                    with open(signals_cache_file, "rb") as f: signals = pickle.load(f)
                    print("\n✓ Successfully loaded prices and signals from cache!")
                except Exception as e:
                    print(f"Cache read failed: {e}. Re-generating.")
            
            if full_price_data is None or signals is None:
                all_tickers = base_cfg['tickers'] + base_cfg['target']
                if base_cfg['benchmark'] not in all_tickers: all_tickers.append(base_cfg['benchmark'])
                full_price_data = get_initial_price_data(all_tickers)
                if full_price_data.empty: return

                signals = generate_signals(
                    unique(base_cfg['tickers'] + base_cfg['target']), full_price_data, base_cfg['signal_types'],
                    base_cfg.get('rsi_periods'), base_cfg.get('price_sma_periods'),
                    base_cfg.get('price_ema_periods'), base_cfg.get('returns_ma_periods')
                )
                try:
                    with open(prices_cache_file, "wb") as f: pickle.dump(full_price_data, f)
                    with open(signals_cache_file, "wb") as f: pickle.dump(signals, f)
                    print("✓ Caching complete.")
                except Exception as e:
                    print(f"Error saving to cache: {e}")
            
            blackout_ranges = get_blackout_ranges_from_user()

        # --- Apply Blackouts ---
        if blackout_ranges:
            print(f"\nApplying blackout windows: {blackout_ranges}")
            full_price_data = apply_blackout_ranges(full_price_data, blackout_ranges)

        print("\n" + "="*20 + " STARTING EVALUATION " + "="*20)
        
        # --- The Stateful Evaluation Loop ---
        while True:
            # Determine which evaluation methods are in the original plan
            original_plan = base_cfg.get('eval_modes', [])
            
            # Find the next task that has NOT been completed
            next_mode_to_run = None
            for mode in original_plan:
                if mode not in completed_evaluations:
                    next_mode_to_run = mode
                    break

            # If all tasks are completed, exit the loop
            if next_mode_to_run is None:
                print("\nAll evaluation methods from the original plan are complete.")
                break

            # Set the configuration for this specific pass
            eval_modes = [next_mode_to_run]
            eval_config = base_cfg.get('eval_config', EvaluationConfig())
            
            # --- Run index based on completed tasks ---
            run_idx = len(completed_evaluations) + 1

            # --- Naming Convention Logic ---
            method_name_short = next_mode_to_run.value
            outdir = parent_dir / f"run_{run_idx}_{method_name_short}"
            os.makedirs(outdir, exist_ok=True)
            paths = RunPaths(str(outdir))
            
            print("\n============================================================")
            print(f"STARTING/RESUMING EVALUATION RUN #{run_idx} ({method_name_short})")
            print(f"Results will be saved to: {outdir}")

            # This call will now run ONE method and save its results in the correct subfolder
            all_results, completed_evaluations = run_comprehensive_evaluation(
                signals, full_price_data, base_cfg['target'],
                eval_modes, eval_config, paths,
                name_prefix="enhanced_signals", base_cfg=base_cfg,
                completed_evaluations=completed_evaluations
            )
            
            # Re-save the master checkpoint with the updated bookmark list
            try:
                master_data_to_save = {
                    'base_cfg': base_cfg, 'signals': signals, 'full_price_data': full_price_data,
                    'blackout_ranges': blackout_ranges, 'run_idx': run_idx,
                    'completed_evaluations': completed_evaluations
                }
                with open(master_checkpoint_path, 'wb') as f:
                    pickle.dump(master_data_to_save, f)
                print(f"✓ Master checkpoint UPDATED for project '{parent_dir.name}'.")
            except Exception as e:
                print(f"CRITICAL WARNING: Could not save master checkpoint! Error: {e}")

        # --- Final Reporting ---
        _cleanup_tqdm()
        if os.path.exists(master_checkpoint_path):
            os.remove(master_checkpoint_path)
        print("\nFull analysis complete. Master checkpoint removed.")

    except KeyboardInterrupt:
        safe_print("\nBacktesting interrupted. Progress may be saved.")
    except Exception as e:
        safe_print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

# Import the essential functions from the original script that we need
def convert_signal_to_composer_format(signal_name, target_ticker, safe_asset="BIL"):
    """Convert backtesting signal name to Composer format."""
    if not COMPOSER_TOOLS_AVAILABLE:
        return None

    try:
        if '+' in signal_name:
            individual_signals = signal_name.split('+')
            composer_condition = '__'.join(individual_signals)
        else:
            composer_condition = signal_name

        composer_code = generate_symphony_code(composer_condition, target_ticker, safe_asset)
        return composer_code

    except Exception as e:
        print(f"Warning: Could not convert signal '{signal_name}' to Composer format: {e}")
        return None

# === CORRECTED VERSION of _sanitize_returns ===
def _sanitize_returns(r) -> pd.Series:
    """
    Ensure numeric float returns with no NaNs/inf, returning a pd.Series.
    This version is corrected to always return a Series, allowing chained commands.
    """
    # If it's already a Series, we can work on it directly for efficiency
    if isinstance(r, pd.Series):
        s = r
    else:
        # If it's an array, list, or something else, convert it to a Series
        s = pd.Series(r)

    # Coerce any non-numeric types (like objects or errors) to NaN
    s = pd.to_numeric(s, errors='coerce')
    
    # Replace infinite values with NaN and then fill all NaNs with 0.0
    s = s.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    
    # Ensure the data type is a standard float
    return s.astype('float64')

def calculate_robustness_score(train_metric, test_metric):
    """Calculate robustness score comparing train vs test performance"""
    # Negative => zero robustness; zero (flat) => small penalty
    if train_metric < 0 or test_metric < 0:
        return 0.0
    if train_metric == 0 or test_metric == 0:
        return 0.0  # or small epsilon if you prefer
    ratio = min(test_metric / train_metric, train_metric / test_metric)
    return float(ratio)

def apply_comprehensive_filter(results, tim_min=0.025, mdd_max=-0.75, quant_filter=0.66, robustness_cutoff=0.0):
    """Apply comprehensive filtering using all user-specified parameters"""
    if len(results) == 0:
        return results

    initial_count = len(results)
    filtered_results = results.copy()

    # 1. Time in Market filter
    if tim_min > 0:
        filtered_results = filtered_results[filtered_results['Time in Market'] >= tim_min]
        print(f"   Time in Market filter (≥{tim_min:.1%}): {len(filtered_results)} signals")

    # 2. Max Drawdown filter
    if mdd_max < 0:
        filtered_results = filtered_results[filtered_results['Max Drawdown'] >= mdd_max]
        print(f"   Max Drawdown filter (≥{mdd_max:.1%}): {len(filtered_results)} signals")

    # 3. Quantile filter (keep top performers)
    if quant_filter > 0 and quant_filter < 1:
        if 'Smart Sharpe' in filtered_results.columns:
            # Sort by Smart Sharpe and keep top quantile
            filtered_results = filtered_results.sort_values('Smart Sharpe', ascending=False)
            keep_count = max(1, int(len(filtered_results) * (1 - quant_filter)))
            filtered_results = filtered_results.head(keep_count)
            print(f"   Quantile filter (top {1-quant_filter:.1%}): {len(filtered_results)} signals")

    # 4. Robustness filter
    if robustness_cutoff > 0 and 'Robustness_Score' in filtered_results.columns:
        filtered_results = filtered_results[filtered_results['Robustness_Score'] >= robustness_cutoff]
        print(f"   Robustness filter (≥{robustness_cutoff:.2f}): {len(filtered_results)} signals")

    final_count = len(filtered_results)
    print(f"   Comprehensive filtering: {initial_count} → {final_count} signals")

    return filtered_results

def apply_robustness_filter(results, robustness_cutoff):
    """Apply robustness cutoff filter to results (legacy function for backward compatibility)"""
    if robustness_cutoff <= 0 or 'Robustness_Score' not in results.columns:
        return results

    initial_count = len(results)
    filtered_results = results[results['Robustness_Score'] >= robustness_cutoff]
    filtered_count = len(filtered_results)

    return filtered_results

def generate_composer_output(results, top_n=10, safe_asset="BIL", result_type="test", robustness_cutoff=0.0):
    """Generate Composer-ready code for top performing signals."""
    if not COMPOSER_TOOLS_AVAILABLE:
        print("Composer-tools not available. Install with: pip install composer-tools")
        return None

    filtered_results = apply_robustness_filter(results, robustness_cutoff)

    if len(filtered_results) == 0:
        print(f"No signals passed robustness filter (>{robustness_cutoff:.2f})")
        return None

    composer_signals = []

    for idx, row in filtered_results.head(top_n).iterrows():
        signal_name = row['Signal']
        target_ticker = row['Target_Ticker'] if 'Target_Ticker' in row else row['Ticker']

        composer_code = convert_signal_to_composer_format(signal_name, target_ticker, safe_asset)

        if composer_code:
            signal_info = {
                'rank': len(composer_signals) + 1,
                'signal_name': signal_name,
                'target_ticker': target_ticker,
                'performance_metric': row.get('Smart Sharpe', row.get('Sortino Ratio', 0)),
                'total_return': row['Total Return'],
                'max_drawdown': row['Max Drawdown'],
                'composer_code': composer_code,
                'result_type': result_type
            }

            if 'Robustness_Score' in row:
                signal_info['robustness_score'] = row['Robustness_Score']

            composer_signals.append(signal_info)

    return composer_signals

def save_composer_signals(composer_signals, filename='composer_signals.txt'):
    """Save Composer signals to a text file for easy copy-pasting."""
    if not composer_signals:
        return

    try:
        with open(filename, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPOSER TRADING SIGNALS - ENHANCED EVALUATION RESULTS\n")
            f.write("="*80 + "\n\n")

            for signal in composer_signals:
                f.write(f"RANK #{signal['rank']} - {signal['signal_name']}\n")
                f.write(f"Target: {signal['target_ticker']}\n")
                f.write(f"Performance: {signal['performance_metric']:.4f}\n")
                f.write(f"Total Return: {signal['total_return']:.3f}\n")
                f.write(f"Max Drawdown: {signal['max_drawdown']:.3f}\n")

                if 'robustness_score' in signal:
                    f.write(f"Robustness Score: {signal['robustness_score']:.3f}\n")

                f.write("-" * 80 + "\n")
                f.write("COMPOSER CODE (copy-paste ready):\n\n")
                f.write(signal['composer_code'])
                f.write("\n\n" + "="*80 + "\n\n")

        print(f"Composer signals saved to: {filename}")

    except Exception as e:
        print(f"Error saving Composer signals: {e}")

def display_composer_preview(composer_signals, show_top=3):
    """Display a preview of Composer signals in the console."""
    if not composer_signals:
        print("No Composer signals to display.")
        return

    print(f"\n{'='*80}")
    print(f"COMPOSER SIGNALS PREVIEW - ENHANCED EVALUATION (Top {show_top})")
    print(f"{'='*80}")

    for signal in composer_signals[:show_top]:
        print(f"\nRANK #{signal['rank']} - {signal['signal_name']}")
        print(f"Target: {signal['target_ticker']} | Performance: {signal['performance_metric']:.4f}")
        print(f"Return: {signal['total_return']:.3f} | Max DD: {signal['max_drawdown']:.3f}")

        if 'robustness_score' in signal:
            print(f"Robustness Score: {signal['robustness_score']:.3f}")

        print("-" * 80)
        print("COMPOSER CODE:")
        print(signal['composer_code'])
        print("="*80)

    if len(composer_signals) > show_top:
        print(f"\n... and {len(composer_signals) - show_top} more signals saved to file.")

# Import remaining essential functions from original script
def report_data_availability(close_prices: pd.DataFrame, label: str = "Universe"):
    """
    Print first valid date per ticker, the earliest common start (overlap),
    and which ticker(s) limit backtest length.
    """
    if isinstance(close_prices, pd.Series):
        close_prices = close_prices.to_frame()

    print(f"\n=== Data Availability Report ({label}) ===")
    firsts = {}
    for t in close_prices.columns:
        idx = close_prices[t].first_valid_index()
        firsts[t] = idx

    # Pretty print per-ticker starts (ascending by date)
    for t, dt in sorted(firsts.items(), key=lambda kv: (pd.Timestamp.max if kv[1] is None else kv[1])):
        msg_date = "no data" if pd.isna(dt) or dt is None else dt.strftime("%Y-%m-%d")
        print(f"  {t:>8}: {msg_date}")

    # Overlap start = latest of the first-valid dates that exist
    valid_dates = [dt for dt in firsts.values() if pd.notna(dt)]
    if not valid_dates:
        print("No valid data for any ticker.")
        return None, []

    overlap_start = max(valid_dates)
    limiters = [t for t, dt in firsts.items() if pd.notna(dt) and dt == overlap_start]

    print(f"\nEarliest common start (all tickers): {overlap_start.strftime('%Y-%m-%d')}")
    print(f"Limiting ticker(s): {', '.join(limiters)}")
    return overlap_start, limiters

# === NEW, EFFICIENT DATA GATHERING FUNCTION ===
def get_initial_price_data(tickers: list) -> pd.DataFrame:
    """
    Downloads max history for all tickers ONCE, identifies the optimal
    start date, trims the data to that date, and returns the final,
    ready-to-use DataFrame.
    """
    print("\n=== Initial Data Acquisition ===")
    
    # 1. Download all the data just one time
    safe_print(f"Downloading MAX history for {len(tickers)} tickers...")
    full_history = download_prices_max_debug(tickers)
    
    # Make the DataFrame's index timezone-naive before slicing to prevent errors
    full_history.index = full_history.index.tz_localize(None)

    # 2. Find the common start date from the data we just downloaded
    print("\nDetermining optimal start date from downloaded data...")
    overlap_start, _ = report_data_availability(full_history, label="Full MAX Pull")
    
    if overlap_start is None:
        print("ERROR: No common date range found for tickers. Exiting.")
        return pd.DataFrame()
        
    print(f"✓ Optimal (overlap) start date: {overlap_start.strftime('%Y-%m-%d')}")
    
    # 3. Prepare and trim the DataFrame
    start_timestamp = pd.Timestamp(overlap_start)
    

    
    # Trim the data to the common start date and drop any remaining rows that have NaNs
    trimmed_data = full_history.loc[start_timestamp:].dropna(how='any')
    
    if trimmed_data.empty:
        print("Error: Data is empty after trimming to the overlap start date.")
        return pd.DataFrame()
    
    years_span = (trimmed_data.index[-1] - trimmed_data.index[0]).days / 365.25
    print(f"✓ Data successfully downloaded and trimmed to {len(trimmed_data)} days ({years_span:.1f} years).")
    return trimmed_data

# === FINAL, SURGICALLY-FIXED VERSION of calculate_quantstats_metrics ===
def calculate_quantstats_metrics(returns, benchmark_returns=None):
    """
    This version replaces the buggy quantstats functions ('calmar', 'max_drawdown')
    with manual, robust calculations to permanently fix the Timedelta error.
    """
    try:
        # Step 1: Sanitize the input into a clean pandas Series.
        clean_returns_series = _sanitize_returns(returns)

        # Step 2: Check for an empty or flat series.
        if len(clean_returns_series) < 2 or clean_returns_series.std() == 0:
            return { 'Total Return': 0, 'Sharpe Ratio': 0, 'Smart Sharpe': 0, 'Sortino Ratio': 0, 'Calmar Ratio': 0, 'Max Drawdown': 0, 'VaR (95%)': 0, 'CVaR (95%)': 0, 'Volatility': 0, 'Skewness': 0, 'Kurtosis': 0, 'Win Rate': 0, 'Best Day': 0, 'Worst Day': 0, 'Avg Win': 0, 'Avg Loss': 0 }

        # Create a clean Series with a simple integer index for the working functions
        final_returns = pd.Series(list(clean_returns_series.values))
        
        # === METRIC CALCULATIONS ===
        metrics = {}
        
        # --- Manual, Safe Calculations for Problematic Metrics ---
        
        # 1. Total Return
        cum_returns = (1 + final_returns).cumprod()
        metrics['Total Return'] = cum_returns.iloc[-1] - 1 if len(cum_returns) > 0 else 0

        # 2. Max Drawdown (manual calculation)
        high_water_mark = cum_returns.cummax()
        drawdown = (cum_returns - high_water_mark) / high_water_mark
        metrics['Max Drawdown'] = drawdown.min() if not drawdown.empty else 0
        
        # 3. Calmar Ratio (manual calculation using our safe max_drawdown)
        # Note: A true Calmar uses annualized return. For robustness and to avoid date math,
        # we will use the simple total return. This is consistent and error-free.
        if metrics['Max Drawdown'] != 0:
            metrics['Calmar Ratio'] = metrics['Total Return'] / abs(metrics['Max Drawdown'])
        else:
            metrics['Calmar Ratio'] = 0.0

        # --- Use quantstats for the functions that we know work correctly ---
        metrics['Sharpe Ratio'] = qs.stats.sharpe(final_returns)
        metrics['Sortino Ratio'] = qs.stats.sortino(final_returns)
        metrics['Volatility'] = qs.stats.volatility(final_returns)
        
        try:
            metrics['Smart Sharpe'] = qs.stats.smart_sharpe(final_returns)
        except Exception:
            skew = qs.stats.skew(final_returns)
            kurt = qs.stats.kurtosis(final_returns)
            sharpe = metrics.get('Sharpe Ratio', 0)
            metrics['Smart Sharpe'] = sharpe * (1 + (skew/6) * sharpe - ((kurt-3)/24) * sharpe**2)

        metrics['VaR (95%)'] = qs.stats.var(final_returns)
        metrics['CVaR (95%)'] = qs.stats.cvar(final_returns)
        metrics['Skewness'] = qs.stats.skew(final_returns)
        metrics['Kurtosis'] = qs.stats.kurtosis(final_returns)
        
        positive_returns = final_returns[final_returns > 0]
        negative_returns = final_returns[final_returns < 0]

        metrics['Win Rate'] = len(positive_returns) / len(final_returns[final_returns != 0]) if len(final_returns[final_returns != 0]) > 0 else 0
        metrics['Best Day'] = final_returns.max()
        metrics['Worst Day'] = final_returns.min()
        metrics['Avg Win'] = positive_returns.mean() if len(positive_returns) > 0 else 0
        metrics['Avg Loss'] = negative_returns.mean() if len(negative_returns) > 0 else 0
        
        return metrics

    except Exception as e:
        print(f"CATASTROPHIC failure in calculate_quantstats_metrics: {e}")
        return { 'Total Return': 0, 'Sharpe Ratio': 0, 'Smart Sharpe': 0, 'Sortino Ratio': 0, 'Calmar Ratio': 0, 'Max Drawdown': 0, 'VaR (95%)': 0, 'CVaR (95%)': 0, 'Volatility': 0, 'Skewness': 0, 'Kurtosis': 0, 'Win Rate': 0, 'Best Day': 0, 'Worst Day': 0, 'Avg Win': 0, 'Avg Loss': 0 }

# === HELPER FUNCTION FOR PARALLEL BACKTESTING (Must be at top-level) ===
# === FINAL CORRECTED VERSION of _run_single_backtest ===
def _run_single_backtest(args: tuple) -> dict:
    """
    Helper function for parallel backtesting. Applies precond_mask per-slice
    inside the worker so masking is correct within each windowed evaluation.
    """
    (signal_name, signal, price_data, target_ticker, daily_ret, precond_mask, EXECUTION_MODE) = args

    import numpy as np
    import pandas as pd

    def align_signal_and_returns(signal, returns):
        if EXECUTION_MODE == "MOC":
            return signal, returns.shift(-1).fillna(0.0)
        else:  # NEXT_BAR
            return signal.shift(1).fillna(False), returns

    sig = signal.reindex(price_data.index).fillna(False)
    if precond_mask is not None:
        pc = precond_mask.reindex(price_data.index).fillna(False)
        sig = sig & pc
    
    sig, aligned_returns = align_signal_and_returns(sig, daily_ret[target_ticker])
    ret = (sig * aligned_returns).astype('float64')
    tim = sig.mean()

    metrics = calculate_quantstats_metrics(ret)
    
    metrics.update({
        'Signal': signal_name,
        'Ticker': target_ticker,
        'Time in Market': tim,
        'Signal Returns': ret
    })

    return metrics


# === FINAL PARALLEL VERSION of backtest_signals ===
def backtest_signals(signals, price_data, target_tickers, benchmark_data=None, period_name="", precond_mask=None):
    """Backtest individual signals using a ProcessPoolExecutor for parallel computation."""
    daily_ret = price_data.pct_change().fillna(0.0)

    # Prepare all the tasks to be run in parallel.
    tasks = []
    for target_ticker in target_tickers:
        for signal_name, signal in signals.items():
            task_args = (signal_name, signal, price_data, target_ticker,
                         daily_ret, precond_mask, EXECUTION_MODE)
            tasks.append(task_args)

    desc = f'Backtesting {period_name} signals' if period_name else 'Backtesting signals'
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results_iterator = executor.map(_run_single_backtest, tasks)
        results = list(tqdm(results_iterator, total=len(tasks), desc=desc, leave=False))

    return pd.DataFrame(results)

# === NEW, FASTER MERGE FUNCTION ===
def merge_train_test_results(train_results, test_results, sort_by):
    """
    Merge train and test results using an optimized, vectorized pandas merge.
    This is much faster than the original loop-based approach.
    """
    if train_results.empty or test_results.empty:
        print("Warning: No results to merge")
        return pd.DataFrame(), pd.DataFrame()

    # Prepare the training results: keep only essential metrics and rename them.
    # We do not need the 'Signal Returns' from the training set.
    train_cols_to_rename = {
        'Total Return': 'Train_Total_Return',
        'Smart Sharpe': 'Train_Smart_Sharpe',
        'Sharpe Ratio': 'Train_Sharpe_Ratio',
        'Sortino Ratio': 'Train_Sortino_Ratio',
        'Calmar Ratio': 'Train_Calmar_Ratio',
        'Max Drawdown': 'Train_Max_Drawdown'
    }
    
    # Create a clean subset of training data for the merge.
    train_subset = train_results[['Signal', 'Ticker'] + list(train_cols_to_rename.keys())].copy()
    train_subset.rename(columns=train_cols_to_rename, inplace=True)

    # Perform a single, highly optimized merge operation.
    # This finds all rows where 'Signal' and 'Ticker' match in both tables.
    merged_df = pd.merge(test_results, train_subset, on=['Signal', 'Ticker'], how='inner')

    if merged_df.empty:
        # This can happen if no signals appear in both train and test sets
        return pd.DataFrame(), pd.DataFrame()

    # Calculate robustness scores in a vectorized way (no loops = very fast)
    # The .get() method is used to safely access columns that might not exist in combo results.
    robustness_sharpe = merged_df.apply(
        lambda row: calculate_robustness_score(row.get('Train_Smart_Sharpe'), row.get('Smart Sharpe')), axis=1)
    robustness_return = merged_df.apply(
        lambda row: calculate_robustness_score(row.get('Train_Total_Return'), row.get('Total Return')), axis=1)
    robustness_sortino = merged_df.apply(
        lambda row: calculate_robustness_score(row.get('Train_Sortino_Ratio'), row.get('Sortino Ratio')), axis=1)
    
    merged_df['Robustness_Sharpe'] = robustness_sharpe
    merged_df['Robustness_Return'] = robustness_return
    merged_df['Robustness_Sortino'] = robustness_sortino
    merged_df['Robustness_Score'] = (robustness_sharpe + robustness_return + robustness_sortino) / 3.0

    print(f"RANKING BY OUT-OF-SAMPLE (TEST) PERFORMANCE: {sort_by}")
    merged_df = merged_df.sort_values(by=sort_by, ascending=False)
    robust_df = merged_df.sort_values(by='Robustness_Score', ascending=False)

    return merged_df, robust_df

def get_user_inputs():
    """Get user inputs for backtesting parameters - original function"""
    print("=== Signal Backtesting Configuration with QuantStats ===")

    # Get target tickers
    print("Target Tickers (tickers to invest in):")
    print("Example: TECL,QQQ,BIL")
    target_input = safe_input("Enter target tickers (comma-separated): ").strip()
    if not target_input:
        target_input = "TECL,QQQ,BIL"
        print(f"Using default: {target_input}")
    target = [t.strip().upper() for t in target_input.split(',') if t.strip()]

    # Get reference tickers
    print("\nReference Tickers (other assets to crosscheck signals with):")
    print("Example: KMLM,CORP,FDN,XLU,XLK,SPY")
    tickers_input = safe_input("Enter reference tickers (comma-separated): ").strip()
    if not tickers_input:
        tickers_input = "KMLM,CORP,FDN,XLU,XLK,SPY"
        print(f"Using default: {tickers_input}")
    tickers = [t.strip().upper() for t in tickers_input.split(',') if t.strip()]

    # Get benchmark ticker
    print("\nBenchmark Ticker (for risk-adjusted metrics):")
    benchmark_input = safe_input("Enter benchmark ticker [SPY]: ").strip()
    benchmark = benchmark_input.upper() if benchmark_input else "SPY"

    # Get safe asset for Composer signals
    if COMPOSER_TOOLS_AVAILABLE:
        print("\nSafe Asset (for Composer 'else' condition):")
        safe_asset_input = safe_input("Enter safe asset ticker [BIL]: ").strip()
        safe_asset = safe_asset_input.upper() if safe_asset_input else "BIL"
    else:
        safe_asset = "BIL"

    # Get signal type
    print("\nAvailable Signal Types:")
    print("1. RSI - Relative Strength Index signals")
    print("2. CUMRET - Cumulative Return signals")
    print("3. RETURNS_MA - Moving Average of returns signals")
    print("4. STD - Standard Deviation of returns signals")
    print("5. PRICE_SMA - Price SMA signals (price vs SMA, SMA vs SMA)")
    print("6. PRICE_EMA - Price EMA signals (price vs EMA, EMA vs EMA)")
    print("7. ALL - All signal types (slower)")
    print("8. CUSTOM - Select multiple signal families (e.g., RSI, PRICE_SMA)")

    signal_choice = safe_input("Select signal type (1-8) [1]: ").strip()
    if not signal_choice:
        signal_choice = "1"
        print("Using default: RSI signals")

    signal_types = {
        '1': ['RSI'],
        '2': ['CUMRET'],
        '3': ['RETURNS_MA'],  # renamed from 'MA'
        '4': ['STD'],
        '5': ['PRICE_SMA'],
        '6': ['PRICE_EMA'],
        '7': ['RSI', 'CUMRET', 'RETURNS_MA', 'STD', 'PRICE_SMA', 'PRICE_EMA']
    }

    # Handle custom multi-select option
    if signal_choice == "8":
        print("\nCustom Signal Family Selection:")
        print("Available families: RSI, CUMRET, RETURNS_MA, STD, PRICE_SMA, PRICE_EMA")
        print("Example: RSI, PRICE_SMA")
        print("This will generate both RSI signals (RSI_10_TQQQ_LT_40) and price vs SMA signals (PRICE_TQQQ_GT_SMA_50_TQQQ)")

        SUPPORTED = {'RSI','RETURNS_MA','CUMRET','STD','PRICE_SMA','PRICE_EMA'}
        raw = safe_input("Enter signal families (comma-separated): ").strip()

        if raw:
            selected_signals = [s.strip().upper() for s in raw.split(',') if s.strip().upper() in SUPPORTED]
            if not selected_signals:
                print("No valid signal families found. Using default: RSI")
                selected_signals = ['RSI']
            else:
                print(f"Selected signal families: {selected_signals}")
        else:
            print("No input provided. Using default: RSI")
            selected_signals = ['RSI']
    else:
        selected_signals = signal_types.get(signal_choice, ['RSI'])

    # RSI-specific configuration
    rsi_periods = None
    if 'RSI' in selected_signals:
        print("\nRSI Configuration:")
        print("1. Default RSI periods (5, 10, 15, 20, 25, 30)")
        print("2. Custom RSI period combinations")

        rsi_choice = safe_input("Select RSI configuration (1-2) [1]: ").strip()

        if rsi_choice == "2":
            print("\nCustom RSI Period Configuration:")
            rsi_input = safe_input("Enter RSI period combinations [default]: ").strip()

            if rsi_input:
                try:
                    rsi_periods = []
                    for combo in rsi_input.split(';'):
                        if ',' in combo:
                            p1, p2 = combo.split(',')
                            rsi_periods.append((int(p1.strip()), int(p2.strip())))
                        else:
                            p = int(combo.strip())
                            rsi_periods.append((p, p))

                    print(f"Custom RSI periods configured: {rsi_periods}")

                except ValueError:
                    print("Invalid RSI period format. Using default periods.")
                    rsi_periods = None
            else:
                print("Using default RSI periods.")
                rsi_periods = None

    # Custom period configuration for other signal types
    price_sma_periods = None
    price_ema_periods = None
    returns_ma_periods = None

    if 'PRICE_SMA' in selected_signals:
        price_sma_periods = _parse_periods("SMA day-counts", [20, 30, 50, 100, 150, 200])

    if 'PRICE_EMA' in selected_signals:
        price_ema_periods = _parse_periods("EMA day-counts", [20, 30, 50, 100, 150, 200])

    if 'RETURNS_MA' in selected_signals:
        returns_ma_periods = _parse_periods("RETURNS_MA day-counts", list(range(10, 110, 10)))

    # Sorting preference
    print(f"\nSelected signal types: {selected_signals}")
    print("\nSorting Options:")
    print("1. Smart Sharpe - Quantstats Smart Sharpe ratio")
    print("2. Sharpe Ratio - Traditional Sharpe ratio")
    print("3. Sortino Ratio - Downside deviation adjusted")
    print("4. Calmar Ratio - Total return / Max Drawdown")
    print("5. Total Return - Absolute returns")
    print("6. Robustness Score - Train/Test consistency metric")

    sort_choice = safe_input("Select sorting metric (1-6) [1]: ").strip()
    sort_options = {
        '1': 'Smart Sharpe',
        '2': 'Sharpe Ratio',
        '3': 'Sortino Ratio',
        '4': 'Calmar Ratio',
        '5': 'Total Return',
        '6': 'Robustness_Score'
    }
    sort_by = sort_options.get(sort_choice, 'Smart Sharpe')
    print(f"Sorting by: {sort_by}")

    # Get other parameters with defaults
    print("\nOptional Parameters (press Enter for defaults):")

    tim = safe_input("Time in Market minimum (e.g., 0.025 = 2.5%) [0.025]: ").strip()
    tim = float(tim) if tim else 0.025

    mdd = safe_input("Max Drawdown Maximum (e.g., -0.75 = 75% MDD) [-0.75]: ").strip()
    mdd = float(mdd) if mdd else -0.75

    quant = safe_input("Quantile filter (0.66 = drop bottom 66%) [0.66]: ").strip()
    quant = float(quant) if quant else 0.66

    return {
        'target': target,
        'tickers': tickers,
        'benchmark': benchmark,
        'safe_asset': safe_asset,
        'signal_types': selected_signals,
        'sort_by': sort_by,
        'tim': tim,
        'mdd': mdd,
        'quant': quant,
        'rsi_periods': rsi_periods,
        'price_sma_periods': price_sma_periods,
        'price_ema_periods': price_ema_periods,
        'returns_ma_periods': returns_ma_periods
    }

def _combine_series(a: pd.Series, b: pd.Series, op: str) -> pd.Series:
    """Combine two signal series using specified boolean operation"""
    if op == "AND":         return (a & b)
    if op == "A_AND_NOT_B": return (a & (~b))
    if op == "B_AND_NOT_A": return ((~a) & b)
    if op == "OR":          return (a | b)
    raise ValueError(op)

def _combine_many(series_list, ops_list):
    """Combine multiple signal series using a list of operations"""
    cur = series_list[0]
    for s, op in zip(series_list[1:], ops_list):
        cur = _combine_series(cur, s, op)
    return cur

def _get_metric(df, name, col, default=-np.inf, ticker=None):
    """Helper function for cleaner and faster metric lookups"""
    try:
        if isinstance(df.index, pd.MultiIndex):
            # Cross-ticker mode: use (signal, ticker) tuple
            if ticker is not None:
                result = df.at[(name, ticker), col]
            else:
                # Fallback: try to find any match for this signal name
                mask = df.index.get_level_values(0) == name
                if mask.any():
                    result = df.loc[mask, col].iloc[0]
                else:
                    return default
        else:
            # Single-ticker mode: use signal name only
            result = df.at[name, col]
        return result
    except Exception:
        return default

def _greedy_build_combo(a_name, signals, partners, ops, train_data, test_data, tkr,
                        tr_map, te_map, sort_by, max_legs, min_train_gain, min_test_gain,
                        enable_cross_ticker=False, signal_ticker_map=None):
    """Greedy forward-selection builder for multi-leg combinations"""
    best_names   = [a_name]
    best_ops     = []
    best_series  = [signals[a_name]]
    best_train   = _get_metric(tr_map, a_name, sort_by, ticker=signal_ticker_map.get(a_name) if enable_cross_ticker else None)
    best_test    = _get_metric(te_map, a_name, sort_by, ticker=signal_ticker_map.get(a_name) if enable_cross_ticker else None)

    while len(best_names) < max_legs:
        improved = None
        for b_name in partners:
            if b_name in best_names:
                continue
            for op in ops:
                candidate_series = _combine_many(best_series + [signals[b_name]],
                                                 best_ops + [op])
                # backtest train
                tr = candidate_series.reindex(train_data.index).fillna(False)
                tr_returns = train_data.pct_change()[tkr].fillna(0.0)
                tr, tr_aligned_returns = align_signal_and_returns(tr, tr_returns)
                tr_ret = (tr * tr_aligned_returns).astype('float64')
                tr_mx  = calculate_quantstats_metrics(tr_ret)
                if tr_mx.get(sort_by, -np.inf) < (max(best_train, _get_metric(tr_map, b_name, sort_by, ticker=signal_ticker_map.get(b_name) if enable_cross_ticker else None)) + min_train_gain):
                    continue
                # backtest test
                te = candidate_series.reindex(test_data.index).fillna(False)
                te_returns = test_data.pct_change()[tkr].fillna(0.0)
                te, te_aligned_returns = align_signal_and_returns(te, te_returns)
                te_ret = (te * te_aligned_returns).astype('float64')
                te_mx  = calculate_quantstats_metrics(te_ret)
                if te_mx.get(sort_by, -np.inf) < (max(best_test, _get_metric(te_map, b_name, sort_by, ticker=signal_ticker_map.get(b_name) if enable_cross_ticker else None)) + min_test_gain):
                    continue
                # candidate improves; track best
                key_metric = te_mx.get(sort_by, -np.inf)
                if (improved is None) or (key_metric > improved[0]):
                    improved = (key_metric, b_name, op, tr_mx, te_mx, te_ret, candidate_series)

        if improved is None:
            break  # no improving addition; stop growing

        _, b_name, op, tr_mx, te_mx, te_ret, candidate_series = improved
        best_names.append(b_name)
        best_ops.append(op)
        best_series = [signals[n] for n in best_names]  # keep list in sync
        best_train  = tr_mx.get(sort_by, best_train)
        best_test   = te_mx.get(sort_by, best_test)

    if len(best_names) == 1:
        return None  # never improved beyond the seed
    combo_name = best_names[0]
    for op, nm in zip(best_ops, best_names[1:]):
        combo_name += f"+{op}+{nm}"

    # Build final row (like your pairwise path)
    return {
        'Signal': combo_name,
        'Ticker': tkr,
        # Combo_ID is now handled by the built-in superset
        'Total Return': te_mx['Total Return'],
        'Smart Sharpe': te_mx['Smart Sharpe'],
        'Sharpe Ratio': te_mx['Sharpe Ratio'],
        'Sortino Ratio': te_mx['Sortino Ratio'],
        'Calmar Ratio': te_mx['Calmar Ratio'],
        'Max Drawdown': te_mx['Max Drawdown'],
        'VaR (95%)': te_mx['VaR (95%)'],
        'CVaR (95%)': te_mx['CVaR (95%)'],
        'Volatility': te_mx['Volatility'],
        'Skewness': te_mx['Skewness'],
        'Kurtosis': te_mx['Kurtosis'],
        'Win Rate': te_mx['Win Rate'],
        'Time in Market': (te_ret != 0).mean(),
        'Train_Total_Return': tr_mx['Total Return'],
        'Train_Smart_Sharpe': tr_mx['Smart Sharpe'],
        'Train_Sharpe_Ratio': tr_mx['Sharpe Ratio'],
        'Train_Sortino_Ratio': tr_mx['Sortino Ratio'],
        'Train_Calmar_Ratio': tr_mx['Calmar Ratio'],
        'Train_Max_Drawdown': tr_mx['Max Drawdown'],
        'Signal Returns': te_ret,
        'Combo_Op': '+'.join(best_ops),
        'Member_A': best_names[0],
        'Member_B': '|'.join(best_names[1:]),
        'Best_Member_Test': best_test,
        'Best_Member_Train': best_train,
        'Synergy_Test': te_mx.get(sort_by, np.nan) - best_test,
        'Synergy_Train': tr_mx.get(sort_by, np.nan) - best_train,
    }

# === INITIALIZER and HELPER for ROBUST PARALLEL COMBO GENERATION ===

def _init_worker(signals_data, train_df, test_df, tr_map_data, te_map_data, sig_ticker_map_data):
    """
    Initializer for each worker process. Sets up global variables to avoid
    passing large objects for every single task.
    """
    global _worker_signals, _worker_train_data, _worker_test_data
    global _worker_tr_map, _worker_te_map, _worker_signal_ticker_map
    
    _worker_signals = signals_data
    _worker_train_data = train_df
    _worker_test_data = test_df
    _worker_tr_map = tr_map_data
    _worker_te_map = te_map_data
    _worker_signal_ticker_map = sig_ticker_map_data


def _find_combos_for_primary(args: tuple) -> list:
    """
    Finds synergistic combo partners for a single primary signal ('a_name').
    This function now reads large data from global variables set by the initializer.
    """
    # Unpack the SMALLER tuple of arguments
    (a_name, tkr, partners, sort_by, min_train_gain, 
     min_test_gain, ops, max_legs, enable_cross_ticker, EXECUTION_MODE) = args
    
    # Imports and helper functions are still needed inside the worker
    import numpy as np
    import pandas as pd

    # Access the large data objects from the worker's global scope
    signals = _worker_signals
    train_data = _worker_train_data
    test_data = _worker_test_data
    tr_map = _worker_tr_map
    te_map = _worker_te_map
    signal_ticker_map = _worker_signal_ticker_map

    def align_signal_and_returns(signal, returns):
        if EXECUTION_MODE == "MOC":
            return signal, returns.shift(-1).fillna(0.0)
        else: # NEXT_BAR
            return signal.shift(1).fillna(False), returns
            
    def _bt(sig_series: pd.Series, prices: pd.DataFrame, ticker: str) -> pd.Series:
        sig = sig_series.reindex(prices.index).fillna(False)
        daily_returns = prices.pct_change()[ticker].fillna(0.0)
        sig, aligned_returns = align_signal_and_returns(sig, daily_returns)
        return (sig * aligned_returns).astype('float64')

    local_combo_rows = []

    if max_legs <= 2:
        a_series = signals[a_name]
        for b_name in partners:
            if a_name == b_name: continue
            
            b_series = signals[b_name]
            for op in ops:
                combo_name = f"{a_name}+{op}+{b_name}"
                combo_full = _combine_series(a_series, b_series, op)

                train_ret = _bt(combo_full, train_data, tkr)
                train_mx = calculate_quantstats_metrics(train_ret)
                
                a_train = _get_metric(tr_map, a_name, sort_by, ticker=signal_ticker_map.get(a_name))
                b_train = _get_metric(tr_map, b_name, sort_by, ticker=signal_ticker_map.get(b_name))
                best_member_train = max(a_train, b_train)

                if (train_mx.get(sort_by, -np.inf) < best_member_train + min_train_gain):
                    continue

                test_ret = _bt(combo_full, test_data, tkr)
                test_mx = calculate_quantstats_metrics(test_ret)
                
                a_test = _get_metric(te_map, a_name, sort_by, ticker=signal_ticker_map.get(a_name))
                b_test = _get_metric(te_map, b_name, sort_by, ticker=signal_ticker_map.get(b_name))
                best_member_test = max(a_test, b_test)

                if (test_mx.get(sort_by, -np.inf) < best_member_test + min_test_gain):
                    continue
                
                row = {
                    'Signal': combo_name, 'Ticker': tkr,
                    'Total Return': test_mx['Total Return'], 'Smart Sharpe': test_mx['Smart Sharpe'],
                    'Sharpe Ratio': test_mx['Sharpe Ratio'], 'Sortino Ratio': test_mx['Sortino Ratio'],
                    'Calmar Ratio': test_mx['Calmar Ratio'], 'Max Drawdown': test_mx['Max Drawdown'],
                    'VaR (95%)': test_mx['VaR (95%)'], 'CVaR (95%)': test_mx['CVaR (95%)'],
                    'Volatility': test_mx['Volatility'], 'Skewness': test_mx['Skewness'],
                    'Kurtosis': test_mx['Kurtosis'], 'Win Rate': test_mx['Win Rate'],
                    'Time in Market': (test_ret != 0).mean(),
                    'Train_Total_Return': train_mx['Total Return'], 'Train_Smart_Sharpe': train_mx['Smart Sharpe'],
                    'Train_Sharpe_Ratio': train_mx['Sharpe Ratio'], 'Train_Sortino_Ratio': train_mx['Sortino Ratio'],
                    'Train_Calmar_Ratio': train_mx['Calmar Ratio'], 'Train_Max_Drawdown': train_mx['Max Drawdown'],
                    'Signal Returns': test_ret, 'Combo_Op': op, 'Member_A': a_name, 'Member_B': b_name,
                    'Best_Member_Test': best_member_test, 'Best_Member_Train': best_member_train,
                    'Synergy_Test': test_mx.get(sort_by, np.nan) - best_member_test,
                    'Synergy_Train': train_mx.get(sort_by, np.nan) - best_member_train,
                    'Member_A_Ticker': signal_ticker_map.get(a_name, 'Unknown'),
                    'Member_B_Ticker': signal_ticker_map.get(b_name, 'Unknown'),
                    'Is_Cross_Ticker': signal_ticker_map.get(a_name) != signal_ticker_map.get(b_name),
                }
                local_combo_rows.append(row)
    else:
        # This part remains the same for multi-leg combos
        row = _greedy_build_combo(
            a_name, signals, partners, ops, train_data, test_data, tkr,
            tr_map, te_map, sort_by, max_legs, min_train_gain, min_test_gain,
            enable_cross_ticker, signal_ticker_map
        )
        if row is not None:
            local_combo_rows.append(row)
            
    return local_combo_rows


# === NEW ROBUST PARALLEL VERSION of enrich_with_synergistic_combos ===
def enrich_with_synergistic_combos(
    signals: Dict[str, pd.Series],
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_tickers: List[str],
    train_results: pd.DataFrame,
    test_results: pd.DataFrame,
    sort_by: str = "Smart Sharpe",
    K_primary: int = 30,
    M_partner: int = 40,
    ops: Tuple[str, ...] = ("AND", "A_AND_NOT_B", "B_AND_NOT_A", "OR"),
    min_train_gain: float = 0.05,
    min_test_gain: float = 0.00,
    random_state: int = 42,
    show_progress: bool = True,
    progress_leave: bool = False,
    max_legs: int = 2,
    enable_cross_ticker: bool = True,
) -> pd.DataFrame:
    """
    Builds synergistic combos in parallel using an initializer to prevent resource exhaustion.
    """
    rng = np.random.default_rng(random_state)
    
    signal_ticker_map = {row['Signal']: row['Ticker'] for _, row in pd.concat([train_results, test_results]).drop_duplicates(subset=['Signal']).iterrows()}
    
    primaries = train_results.sort_values(sort_by, ascending=False).head(K_primary)["Signal"].tolist()
    all_names = list(signals.keys())
    
    all_combo_rows = []

    for tkr in target_tickers:
        tr_map = train_results.set_index(['Signal', 'Ticker']) if enable_cross_ticker else train_results[train_results["Ticker"] == tkr].set_index("Signal")
        te_map = test_results.set_index(['Signal', 'Ticker']) if enable_cross_ticker else test_results[test_results["Ticker"] == tkr].set_index("Signal")
        
        # Prepare the large, read-only data to be sent to each worker ONCE
        init_args = (signals, train_data, test_data, tr_map, te_map, signal_ticker_map)

        tasks = []
        for a_name in primaries:
            partners = [x for x in all_names if x != a_name]
            if len(partners) > M_partner:
                partners = list(rng.choice(partners, size=M_partner, replace=False))
            
            # The task tuple is now much smaller!
            task_args = (a_name, tkr, partners, sort_by, min_train_gain, 
                         min_test_gain, ops, max_legs, enable_cross_ticker, EXECUTION_MODE)
            tasks.append(task_args)

        desc = f"🔗 Generating combos for {tkr}"
        
        # Create the process pool with the initializer function and its arguments
        with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker, initargs=init_args) as executor:
            results_iterator = executor.map(_find_combos_for_primary, tasks)
            
            list_of_lists = list(tqdm(results_iterator, total=len(tasks), desc=desc, leave=progress_leave))
            
            for sublist in list_of_lists:
                all_combo_rows.extend(sublist)

    if not all_combo_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_combo_rows)
    return df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)

if __name__ == "__main__":
        # === Print startup messages only once from the main process ===
    if COMPOSER_TOOLS_AVAILABLE:
        print("Composer-tools library detected - Composer code generation enabled!")
    if COMBO_MODULES_AVAILABLE:
        print("✓ Built-in combo analysis features enabled - no external modules needed!")
    # =============================================================
    enhanced_main()