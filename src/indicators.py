import pandas as pd
import numpy as np


def calculate_sma(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_ema(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_rsi(series, period):
    """
    Relative Strength Index (Wilder's Smoothing)
    """
    delta = series.diff()

    # Separate gains and losses
    gain = delta.clip(lower=0)
    loss = -1 * delta.clip(upper=0)

    # Wilder's Smoothing uses an EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()

    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Handle edge case where average loss is 0 (RSI = 100)
    rsi = rsi.replace(np.inf, 100)

    # Set the first 'period' rows to NaN to match standard indicator behavior
    rsi.iloc[:period] = np.nan

    return rsi


def calculate_cumret(series, period):
    """Cumulative Return over a rolling window (percentage, not decimal)."""
    return series.pct_change(periods=period) * 100


def calculate_atr(high, low, close, period):
    """
    Wilder's Average True Range.

    high, low, close must be pd.Series aligned to the same DatetimeIndex.
    Uses Wilder's smoothing (EMA with alpha = 1 / period).
    """
    prev_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False).mean()


def calculate_bbands_lower(series, period, num_std=2.0):
    """Lower Bollinger Band: SMA(period) - num_std * rolling_std(period)."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    return sma - (std * num_std)


def calculate_maxdd(series, period):
    """Rolling max drawdown over `period` days (returns positive %, e.g. 15.0 = 15% DD)."""
    def _maxdd(window):
        peak = window.max()
        if peak <= 0:
            return 0.0
        trough = window.min()
        return ((peak - trough) / peak) * 100
    return series.rolling(window=period).apply(_maxdd, raw=True)


def calculate_mareturn(series, period):
    """Rolling mean of daily percentage returns over `period` days."""
    return series.pct_change().rolling(window=period).mean()


def add_indicator(df, asset_role, indicator_name, period):
    """
    Calculates a technical indicator and appends it to the Master DataFrame.
    """
    # Create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    price_col = f"{asset_role}_close"
    if price_col not in df.columns:
        raise ValueError(f"Required price column '{price_col}' not found in DataFrame.")

    indicator_col = f"{asset_role}_{indicator_name}_{period}"

    if indicator_name.upper() == "RSI":
        df[indicator_col] = calculate_rsi(df[price_col], period)
    elif indicator_name.upper() == "SMA":
        df[indicator_col] = calculate_sma(df[price_col], period)
    elif indicator_name.upper() == "EMA":
        df[indicator_col] = calculate_ema(df[price_col], period)
    elif indicator_name.upper() == "CUMRET":
        df[indicator_col] = calculate_cumret(df[price_col], period)
    elif indicator_name.upper() == "ATR":
        high_col = f"{asset_role}_high"
        low_col = f"{asset_role}_low"
        if high_col not in df.columns or low_col not in df.columns:
            raise ValueError(
                f"ATR requires '{high_col}' and '{low_col}' columns in the DataFrame. "
                f"Delete existing ticker CSVs and re-download with OHLC support."
            )
        df[indicator_col] = calculate_atr(df[high_col], df[low_col], df[price_col], period)
    elif indicator_name.upper() == "BBAND_LOWER":
        df[indicator_col] = calculate_bbands_lower(df[price_col], period)
    else:
        raise ValueError(f"Unsupported indicator: {indicator_name}")

    return df
