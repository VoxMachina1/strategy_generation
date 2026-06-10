import pandas as pd
from pathlib import Path


def load_ticker_csv(ticker, data_dir):
    """
    Reads the price CSV for a given ticker.
    Converts 'date' to a pandas datetime object and sorts chronologically.
    Returns df with columns ['date', 'open', 'high', 'low', 'close'] if OHLC present,
    else ['date', 'close'] for legacy CSVs.
    """
    file_path = data_dir / f"{ticker}.csv"
    if not file_path.exists():
        raise FileNotFoundError(f"Data file for {ticker} not found at {file_path}")

    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns:
        return df[['date', 'open', 'high', 'low', 'close']]
    return df[['date', 'close']]


def build_master_dataframe(signal_ticker, target_ticker, benchmark_ticker, data_dir, filter_assets=None):
    """
    Loads signal, target, benchmark, and any global filter CSVs.
    Renames columns according to their roles or exact ticker names.
    Merges them into a single aligned DataFrame, dropping any missing dates.
    """
    if filter_assets is None:
        filter_assets = []

    # 1. Load base 3 assets
    df_signal = load_ticker_csv(signal_ticker, data_dir)
    signal_rename = {'close': 'signal_close'}
    if 'high' in df_signal.columns:
        signal_rename.update({'open': 'signal_open', 'high': 'signal_high', 'low': 'signal_low'})
    df_signal = df_signal.rename(columns=signal_rename)

    df_target = load_ticker_csv(target_ticker, data_dir)
    target_rename = {'close': 'target_close'}
    if 'high' in df_target.columns:
        target_rename.update({'open': 'target_open', 'high': 'target_high', 'low': 'target_low'})
    df_target = df_target.rename(columns=target_rename)

    df_bench = load_ticker_csv(benchmark_ticker, data_dir).rename(columns={'close': 'benchmark_close'})

    # Merge base 3
    master_df = pd.merge(df_signal, df_target, on='date', how='inner')
    master_df = pd.merge(master_df, df_bench, on='date', how='inner')

    # 2. Load and merge filter assets
    for ticker in set(filter_assets):
        df_filter = load_ticker_csv(ticker, data_dir).rename(columns={'close': f'{ticker}_close'})
        master_df = pd.merge(master_df, df_filter, on='date', how='inner')

    # Drop any rows with missing data across all merged assets
    master_df = master_df.dropna().reset_index(drop=True)

    return master_df


def load_multi_ticker_aligned(tickers, data_dir):
    """
    Loads each ticker CSV, extracts the close column, renames it to the ticker symbol.
    Aligns all tickers on date via inner join and sorts chronologically.
    Returns a DataFrame with ticker symbols as columns and a datetime index.

    Args:
        tickers:  list of ticker strings, e.g. ["SPY", "QQQ"]
        data_dir: pathlib.Path to the directory containing {ticker}.csv files

    Returns:
        pd.DataFrame with columns = tickers, index = date (DatetimeIndex),
        sorted ascending by date. Inner join means only dates present for ALL
        tickers are included.
    """
    frames = []
    for ticker in tickers:
        df = load_ticker_csv(ticker, data_dir)
        close_series = df.set_index("date")["close"].rename(ticker)
        frames.append(close_series)
    aligned = pd.concat(frames, axis=1, join="inner")
    aligned = aligned.sort_index()
    return aligned
