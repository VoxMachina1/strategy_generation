# Monte Carlo simulations of forward returns using composer back tests
# version 2: fixed drawdown duration to date mapping, added rolling walk mode
# version 3: updated price fetching to current yfinance package
# prairie@Investor's Collaborative 20250418

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta, date
import os
from matplotlib.gridspec import GridSpec
import requests
from typing import List, Dict
import sys
import subprocess

# Ensure yfinance is installed
try:
    import yfinance
except ImportError:
    print("yfinance not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance

# Set random seed for reproducibility
np.random.seed(42)


def convert_trading_date(date_int):
    """
    Convert trading date integer to datetime object
    """
    date_1 = datetime.strptime("01/01/1970", "%m/%d/%Y")
    dt = date_1 + timedelta(days=int(date_int))
    return dt


class YahooFinanceAPI:
    """Fetches historical price data using the yfinance package."""

    def __init__(self, session=None):
        """
        Initialize the Yahoo Finance API client.

        Args:
            session: Optional requests session (not used with yfinance but kept for compatibility)
        """
        # Try importing yfinance
        try:
            import yfinance as yf

            self.yf = yf
            print("Successfully initialized yfinance package")
        except ImportError:
            print(
                "yfinance package is not installed. Please install it with: pip install yfinance"
            )
            raise ImportError(
                "yfinance package is required to use the Yahoo Finance API"
            )

        # Dictionary to map special tickers to their Yahoo Finance format
        self.ticker_map = {"BRK/B": "BRK-B"}

        # Default settings
        self.rate_limit_delay = 1.0  # seconds between requests
        self.use_batch_download = (
            True  # Set to True to use batch downloading instead of individual downloads
        )
        self.batch_size = 5  # Number of symbols per batch when using batch download

    def fetch_historical_data(
        self, symbols: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.Series]:
        """
        Fetch historical price data for multiple symbols.

        Args:
            symbols: List of ticker symbols to fetch
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping symbols to price series
        """

        print(
            f"Fetching historical data for {len(symbols)} symbols from {start_date} to {end_date}"
        )

        # Map symbols to Yahoo Finance format if needed
        mapped_symbols = {}
        for symbol in symbols:
            yahoo_symbol = self.ticker_map.get(symbol, symbol)
            if yahoo_symbol != symbol:
                print(
                    f"Special ticker handling: Mapping {symbol} to {yahoo_symbol} for Yahoo Finance"
                )
            mapped_symbols[yahoo_symbol] = symbol

        # Choose download method
        if self.use_batch_download and len(symbols) > 1:
            print(f"Using batch download method for {len(symbols)} symbols")
            return self._batch_download(mapped_symbols, start_date, end_date)
        else:
            print(f"Using individual download method for {len(symbols)} symbols")
            return self._individual_download(mapped_symbols, start_date, end_date)

    def _individual_download(
        self, mapped_symbols: Dict[str, str], start_date: str, end_date: str
    ) -> Dict[str, pd.Series]:
        """
        Download data for each symbol individually (more reliable but slower).

        Args:
            mapped_symbols: Dictionary mapping Yahoo symbols to original symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping original symbols to price series
        """
        import time
        import numpy as np

        price_data = {}

        # Process symbols one at a time
        for yahoo_symbol, original_symbol in mapped_symbols.items():
            print(f"Fetching data for {original_symbol}")

            # Try with the primary method
            try:
                # Create a ticker object
                ticker_obj = self.yf.Ticker(yahoo_symbol)

                # Get historical data
                data = ticker_obj.history(
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,  # Use adjusted data
                )

                if data.empty:
                    print(f"No data returned for {original_symbol}")
                    continue

                # Extract the 'Close' prices
                if "Close" in data.columns:
                    series = data["Close"].copy()

                    # Clean the data
                    series = series.dropna()
                    series = series.astype(np.float32)

                    # Make sure the index is timezone naive
                    if series.index.tz is not None:
                        series.index = series.index.tz_convert("America/New_York")
                        series.index = series.index.tz_localize(None)

                    # Remove duplicates
                    series = series[~series.index.duplicated(keep="last")]

                    if not series.empty:
                        series.name = original_symbol  # Use the original symbol name
                        price_data[original_symbol] = series
                        print(
                            f"Successfully retrieved {original_symbol}: {len(series)} points "
                            f"from {series.index[0].strftime('%Y-%m-%d')} "
                            f"to {series.index[-1].strftime('%Y-%m-%d')}"
                        )
                    else:
                        print(f"Series was empty after cleaning for {original_symbol}")
                else:
                    print(f"No Close column found for {original_symbol}")

            except Exception as e:
                print(f"Error fetching data for {original_symbol}: {str(e)}")
                # Try with a modified symbol (some ETFs need this)
                if "-" not in yahoo_symbol and "/" not in yahoo_symbol:
                    mod_symbol = f"{yahoo_symbol}-USD"
                    print(f"Trying alternative symbol format: {mod_symbol}")

                    try:
                        # Create a ticker object with the modified symbol
                        ticker_obj = self.yf.Ticker(mod_symbol)

                        # Get historical data
                        data = ticker_obj.history(
                            start=start_date, end=end_date, auto_adjust=True
                        )

                        if not data.empty and "Close" in data.columns:
                            series = data["Close"].copy()
                            series = series.dropna()
                            series = series.astype(np.float32)

                            # Make sure the index is timezone naive
                            if series.index.tz is not None:
                                series.index = series.index.tz_convert(
                                    "America/New_York"
                                )
                                series.index = series.index.tz_localize(None)

                            # Remove duplicates
                            series = series[~series.index.duplicated(keep="last")]

                            if not series.empty:
                                series.name = (
                                    original_symbol  # Use the original symbol name
                                )
                                price_data[original_symbol] = series
                                print(
                                    f"Successfully retrieved {original_symbol} (as {mod_symbol}): {len(series)} points "
                                    f"from {series.index[0].strftime('%Y-%m-%d')} "
                                    f"to {series.index[-1].strftime('%Y-%m-%d')}"
                                )
                    except Exception as alt_e:
                        print(
                            f"Error with alternative symbol format for {original_symbol}: {str(alt_e)}"
                        )

            # Add a small delay between requests to avoid rate limiting
            time.sleep(self.rate_limit_delay)

        return price_data

    def _batch_download(
        self, mapped_symbols: Dict[str, str], start_date: str, end_date: str
    ) -> Dict[str, pd.Series]:
        """
        Download data for multiple symbols in batches (faster but less reliable).

        Args:
            mapped_symbols: Dictionary mapping Yahoo symbols to original symbols
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            Dict mapping original symbols to price series
        """
        import time
        import pandas as pd
        import numpy as np

        price_data = {}
        yahoo_symbols = list(mapped_symbols.keys())

        # Process symbols in batches
        for i in range(0, len(yahoo_symbols), self.batch_size):
            batch = yahoo_symbols[i : i + self.batch_size]
            print(
                f"Fetching batch of {len(batch)} symbols (batch {i // self.batch_size + 1})"
            )

            try:
                # Download data for the entire batch
                data = self.yf.download(
                    tickers=batch,
                    start=start_date,
                    end=end_date,
                    group_by="ticker",
                    auto_adjust=True,  # Use adjusted data
                    actions=False,  # Don't include dividends
                    progress=False,  # Don't display progress bar
                )

                # Process batch results
                if len(batch) == 1:
                    # Special case for single ticker (different data structure)
                    yahoo_symbol = batch[0]
                    original_symbol = mapped_symbols[yahoo_symbol]

                    if "Close" in data.columns:
                        series = data["Close"].copy()

                        # Clean the data
                        series = series.dropna()
                        series = series.astype(np.float32)

                        # Make sure the index is timezone naive
                        if series.index.tz is not None:
                            series.index = series.index.tz_convert("America/New_York")
                            series.index = series.index.tz_localize(None)

                        # Remove duplicates
                        series = series[~series.index.duplicated(keep="last")]

                        if not series.empty:
                            series.name = original_symbol
                            price_data[original_symbol] = series
                            print(
                                f"Successfully retrieved {original_symbol}: {len(series)} points"
                            )

                else:
                    # Process multiple tickers
                    for yahoo_symbol in batch:
                        original_symbol = mapped_symbols[yahoo_symbol]

                        try:
                            if (
                                yahoo_symbol in data.columns
                                and "Close" in data[yahoo_symbol].columns
                            ):
                                series = data[yahoo_symbol]["Close"].copy()

                                # Clean the data
                                series = series.dropna()
                                series = series.astype(np.float32)

                                # Make sure the index is timezone naive
                                if series.index.tz is not None:
                                    series.index = series.index.tz_convert(
                                        "America/New_York"
                                    )
                                    series.index = series.index.tz_localize(None)

                                # Remove duplicates
                                series = series[~series.index.duplicated(keep="last")]

                                if not series.empty:
                                    series.name = original_symbol
                                    price_data[original_symbol] = series
                                    print(
                                        f"Successfully retrieved {original_symbol}: {len(series)} points"
                                    )
                                else:
                                    print(
                                        f"Empty data for {original_symbol} after cleaning"
                                    )
                            else:
                                print(
                                    f"No data found for {original_symbol} in batch results"
                                )
                        except Exception as e:
                            print(
                                f"Error processing batch data for {original_symbol}: {str(e)}"
                            )

            except Exception as e:
                print(f"Error fetching batch {i // self.batch_size + 1}: {str(e)}")

            # Add delay between batches
            if i + self.batch_size < len(yahoo_symbols):
                time.sleep(self.rate_limit_delay * 2)  # Longer delay between batches

        # Check for failed tickers and retry individually
        failed_symbols = {
            s: mapped_symbols[s]
            for s in mapped_symbols
            if mapped_symbols[s] not in price_data
        }

        if failed_symbols:
            print(f"Retrying {len(failed_symbols)} failed symbols individually...")
            retry_results = self._individual_download(
                failed_symbols, start_date, end_date
            )
            price_data.update(retry_results)

        return price_data


def fetch_backtest(id, start_date, end_date):
    """
    Fetch backtest data from Composer API
    """
    if id.endswith("/details"):
        id = id.split("/")[-2]
    else:
        id = id.split("/")[-1]

    payload = {
        "capital": 100000,
        "apply_reg_fee": True,
        "apply_taf_fee": True,
        "backtest_version": "v2",
        "slippage_percent": 0.0005,
        "start_date": start_date,
        "end_date": end_date,
    }

    url = f"https://backtest-api.composer.trade/api/v2/public/symphonies/{id}/backtest"

    data = requests.post(url, json=payload)
    jsond = data.json()
    symphony_name = jsond["legend"][id]["name"]

    holdings = jsond["last_market_days_holdings"]

    tickers = []
    for ticker in holdings:
        tickers.append(ticker)

    # Extract allocations
    allocations = jsond["tdvm_weights"]
    date_range = pd.date_range(start=start_date, end=end_date)
    df = pd.DataFrame(0.0, index=date_range, columns=tickers)

    for ticker in allocations:
        for date_int in allocations[ticker]:
            trading_date = convert_trading_date(date_int)
            percent = allocations[ticker][date_int]
            df.at[trading_date, ticker] = percent

    return df, symphony_name, tickers


def calculate_portfolio_returns(allocations_df, tickers):
    """
    Calculate daily portfolio returns with properly normalized dates using allocation weighting
    and correct compounding.
    """
    # Find the first row with at least one non-zero value
    first_valid_index = allocations_df[
        (abs(allocations_df) > 0.000001).any(axis=1)
    ].first_valid_index()

    # Get rid of data prior to start of backtest and non-trading days
    allocations_df = allocations_df.loc[(allocations_df != 0).any(axis=1)] * 100.0

    # Add $USD column if not present
    if "$USD" not in allocations_df.columns:
        allocations_df["$USD"] = 0

    # IMPORTANT: Normalize allocation dates to remove time component
    # Convert to datetime and keep only the date part, then convert back to datetime
    allocations_df.index = pd.to_datetime(allocations_df.index).normalize()

    # Extract unique tickers
    unique_tickers = {ticker for ticker in tickers if ticker != "$USD"}

    # Fetch historical prices with adequate buffer
    start_date = allocations_df.index.min() - timedelta(days=10)
    end_date = allocations_df.index.max() + timedelta(days=10)

    print(
        f"Fetching price data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    # Initialize Yahoo Finance API
    yahoo_api = YahooFinanceAPI()

    # Fetch historical prices
    prices_data = yahoo_api.fetch_historical_data(
        list(unique_tickers),
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    )

    # Create price DataFrame
    prices = pd.DataFrame({ticker: prices_data[ticker] for ticker in prices_data})

    # IMPORTANT: Normalize price dates to remove time component
    prices.index = pd.to_datetime(prices.index).normalize()

    # Add $USD column with value 1.0
    prices["$USD"] = 1.0

    # Make sure we have all the tickers
    for ticker in tickers:
        if ticker not in prices.columns and ticker != "$USD":
            print(f"Warning: Price data for {ticker} not found. Setting to NaN.")
            prices[ticker] = np.nan

    # Forward fill missing values
    prices = prices.ffill()
    prices = prices.bfill()
    prices = prices.fillna(1.0)

    # Reorder columns to match tickers
    prices = prices[tickers]

    # Sort DataFrames by index
    allocations_df.sort_index(inplace=True)
    prices.sort_index(inplace=True)

    # Print date info with normalized dates
    price_dates = sorted(prices.index)
    alloc_dates = sorted(allocations_df.index)
    print(
        f"Retrieved {len(price_dates)} price dates from {price_dates[0].date()} to {price_dates[-1].date()}"
    )
    print(
        f"Have {len(alloc_dates)} allocation dates from {alloc_dates[0].date()} to {alloc_dates[-1].date()}"
    )

    # Check if all allocation dates exist in price dates
    missing_dates = set(alloc_dates) - set(price_dates)
    if missing_dates:
        print(
            f"Found {len(missing_dates)} allocation dates without exact price date matches"
        )
        print(
            f"First few missing dates: {[d.date() for d in sorted(missing_dates)[:5]]}"
        )
    else:
        print("All allocation dates have exact matching price dates!")

    # Create a simple dictionary to store price data by date
    price_dict = {}
    for date, row in prices.iterrows():
        date_key = date.strftime(
            "%Y-%m-%d"
        )  # Use string as key for consistent matching
        price_dict[date_key] = row

    # Create dictionaries to store price changes by ticker and date
    price_changes = {}

    # Calculate daily price changes for each ticker
    for ticker in tickers:
        if ticker == "$USD":
            continue  # Skip cash

        price_changes[ticker] = {}
        ticker_prices = prices[ticker]

        # Calculate daily percentage changes
        for i in range(1, len(price_dates)):
            today = price_dates[i]
            yesterday = price_dates[i - 1]

            today_price = ticker_prices.loc[today]
            yesterday_price = ticker_prices.loc[yesterday]

            # Calculate daily percentage change
            if yesterday_price is not None:
                daily_change = ((today_price / yesterday_price) - 1) * 100
                price_changes[ticker][today.strftime("%Y-%m-%d")] = daily_change

    # Initialize daily returns
    daily_returns = pd.Series(index=allocations_df.index[1:], dtype=float)

    # Calculate portfolio returns using the weighted allocation approach
    for i in range(1, len(allocations_df)):
        today_date = allocations_df.index[i]

        today_key = today_date.strftime("%Y-%m-%d")

        # Get yesterday's allocations (these are the active allocations for calculating today's return)
        allocations_yday = allocations_df.iloc[i - 1, :] / 100.0  # Convert to 0-1 range

        # Calculate weighted return for the day
        portfolio_daily_return = 0.0

        for ticker in tickers:
            if ticker == "$USD":
                # Cash has 0% return
                continue

            ticker_allocation = allocations_yday[ticker]

            if ticker_allocation > 0:
                if today_key in price_changes.get(ticker, {}):
                    # Apply allocation weighting to the ticker's return
                    ticker_return = price_changes[ticker][today_key]
                    portfolio_daily_return += ticker_allocation * ticker_return

        # Store the daily return
        daily_returns.iloc[i - 1] = portfolio_daily_return

        # Log information for the last few days
        if i >= len(allocations_df) - 5:
            print(f"\nCalculating return for {today_date.date()}:")
            print(f"Portfolio daily return: {portfolio_daily_return:.4f}%")
            print("Ticker contributions:")

            for ticker in tickers:
                if ticker == "$USD" or allocations_yday[ticker] <= 0:
                    continue

                if today_key in price_changes.get(ticker, {}):
                    ticker_return = price_changes[ticker][today_key]
                    contribution = allocations_yday[ticker] * ticker_return
                    print(
                        f"  {ticker}: {allocations_yday[ticker] * 100:.2f}% allocation, "
                        f"{ticker_return:.4f}% return, {contribution:.4f}% contribution"
                    )

    # Print return statistics
    print("\n--- RETURN STATISTICS ---")
    print(f"Average daily return: {daily_returns.mean():.4f}%")
    print(
        f"Min/Max daily return: {daily_returns.min():.4f}% / {daily_returns.max():.4f}%"
    )
    print(
        f"Positive days: {(daily_returns > 0).sum()} ({(daily_returns > 0).mean() * 100:.2f}%)"
    )

    # Ensure length alignment
    if len(daily_returns) != len(allocations_df.index[1:]):
        print(
            f"Warning: Return length ({len(daily_returns)}) doesn't match allocation dates ({len(allocations_df.index[1:])})"
        )
        # Trim to ensure match
        min_len = min(len(daily_returns), len(allocations_df.index[1:]))
        daily_returns = daily_returns[:min_len]
        dates = allocations_df.index[
            : min_len + 1
        ]  # +1 because first date has no return
    else:
        dates = allocations_df.index

    return daily_returns, dates


def run_monte_carlo_simulation(
    returns, num_simulations=10000, simulation_length=None, annual_periods=252
):
    """Run Monte Carlo simulation using separate sampling for positive and negative returns."""
    if simulation_length is None:
        simulation_length = len(returns)

    returns_array = np.array(returns)

    # Separate positive and negative returns
    positive_returns = returns_array[returns_array > 0]
    negative_returns = returns_array[returns_array <= 0]

    # Calculate probabilities
    prob_positive = len(positive_returns) / len(returns_array)

    print(
        f"Return characteristics - Probability of positive return: {prob_positive:.4f}"
    )
    print(
        f"Positive returns - Mean: {np.mean(positive_returns):.4f}, Std: {np.std(positive_returns):.4f}"
    )
    print(
        f"Negative returns - Mean: {np.mean(negative_returns):.4f}, Std: {np.std(negative_returns):.4f}"
    )

    # Initialize arrays to store simulation results
    cumulative_returns = np.zeros((num_simulations, simulation_length + 1))
    cumulative_returns[:, 0] = 0  # Start with 0% return

    # Arrays to store Sharpe ratios and max drawdowns
    sharpe_ratios = np.zeros(num_simulations)
    max_drawdowns = np.zeros(num_simulations)

    # Arrays to store drawdown durations
    max_drawdown_durations = np.zeros(num_simulations)
    total_drawdown_days = np.zeros(num_simulations)

    # Run simulations
    for i in range(num_simulations):
        # Generate random returns by sampling separately from positive and negative returns
        simulated_returns = np.zeros(simulation_length)
        for j in range(simulation_length):
            if np.random.random() < prob_positive:
                # Sample from positive returns
                simulated_returns[j] = np.random.choice(positive_returns)
            else:
                # Sample from negative returns
                simulated_returns[j] = np.random.choice(negative_returns)

        # Calculate cumulative returns
        cum_return = 0
        cum_returns = [cum_return]
        peak = 0
        max_drawdown = 0

        # Track drawdown durations
        in_drawdown = False
        current_drawdown_duration = 0
        max_dd_duration = 0
        total_dd_days = 0

        for r in simulated_returns:
            # Convert daily percentage return to decimal
            r_decimal = r / 100.0

            # Calculate new cumulative return (compounded)
            cum_return = (1 + cum_return / 100) * (1 + r_decimal) * 100 - 100
            cum_returns.append(cum_return)

            # Update peak and calculate drawdown
            if cum_return > peak:
                peak = cum_return
                # End of drawdown period
                if in_drawdown:
                    in_drawdown = False
                    max_dd_duration = max(max_dd_duration, current_drawdown_duration)
                    current_drawdown_duration = 0

            # Calculate drawdown as a percentage of peak value
            drawdown = ((peak - cum_return) / (1 + peak / 100)) if peak > 0 else 0

            # Track drawdown periods
            if drawdown > 0:
                if not in_drawdown:
                    in_drawdown = True
                if in_drawdown:
                    current_drawdown_duration += 1
                    total_dd_days += 1

            max_drawdown = max(max_drawdown, drawdown)

        # If still in drawdown at end of series, update max_drawdown_duration
        if in_drawdown:
            max_dd_duration = max(max_dd_duration, current_drawdown_duration)

        cumulative_returns[i, :] = cum_returns

        # Calculate Sharpe ratio
        annual_return = cum_return * (annual_periods / simulation_length)
        annual_volatility = np.std(simulated_returns) * np.sqrt(annual_periods)
        sharpe_ratio = (
            annual_return / annual_volatility if annual_volatility != 0 else 0
        )

        sharpe_ratios[i] = sharpe_ratio
        max_drawdowns[i] = max_drawdown
        max_drawdown_durations[i] = max_dd_duration
        total_drawdown_days[i] = total_dd_days

    # Calculate percentiles for paths
    percentile_5 = np.percentile(cumulative_returns, 5, axis=0)
    percentile_25 = np.percentile(cumulative_returns, 25, axis=0)
    percentile_50 = np.percentile(cumulative_returns, 50, axis=0)
    percentile_75 = np.percentile(cumulative_returns, 75, axis=0)
    percentile_95 = np.percentile(cumulative_returns, 95, axis=0)

    # Calculate final returns for all simulations
    final_returns = cumulative_returns[:, -1]

    results = {
        "final_returns": final_returns,
        "paths": cumulative_returns,
        "percentiles": {
            "5": percentile_5,
            "25": percentile_25,
            "50": percentile_50,
            "75": percentile_75,
            "95": percentile_95,
        },
        "sharpe_ratios": sharpe_ratios,
        "max_drawdowns": max_drawdowns,
        "max_drawdown_durations": max_drawdown_durations,
        "total_drawdown_days": total_drawdown_days,
    }

    return results


def analyze_drawdowns(
    returns,
    output_dir,
    period_length,
    test_start_date,
    test_end_date,
    portfolio_name,
    dates=None,
):
    """
    Analyze drawdowns and create visualizations

    Parameters:
    -----------
    returns : list
        List of cumulative returns
    output_dir : str
        Directory to save output files
    period_length : int
        Length of the test period in days
    test_start_date : str
        Start date of the test period
    test_end_date : str
        End date of the test period
    portfolio_name : str
        Name of the portfolio for file naming
    dates : list, optional
        List of date strings corresponding to returns data
    """
    # Create a list of dates if not provided
    if dates is None:
        # Create synthetic dates starting from test_start_date
        start_date = pd.Timestamp(test_start_date)
        dates = [
            (start_date + pd.Timedelta(days=i)).strftime("%Y-%m-%d")
            for i in range(len(returns))
        ]

    # Ensure dates is the same length as returns
    if len(dates) != len(returns):
        print(
            f"Warning: Length mismatch between dates ({len(dates)}) and returns ({len(returns)})"
        )
        # Use the shorter length
        min_length = min(len(dates), len(returns))
        dates = dates[:min_length]
        returns = returns[:min_length]

    # Convert dates to datetime objects for proper comparison
    date_objects = [
        pd.to_datetime(d).date() if isinstance(d, str) else d for d in dates
    ]

    # Calculate drawdown periods and durations
    drawdown_periods = []
    drawdowns = []

    # Track the running peak
    running_peak = returns[0]
    current_drawdown_start_idx = None
    current_drawdown_start_value = None
    in_drawdown = False
    max_drawdown = 0
    max_drawdown_idx = 0

    # Loop through returns to calculate drawdowns and identify periods
    for i, value in enumerate(returns):
        # Update the running peak if we have a new high
        if value > running_peak:
            running_peak = value

            # If we were in a drawdown and now we've reached a new peak, the drawdown is over
            if in_drawdown:
                # Calculate the depth (percentage) of this drawdown period
                # IMPORTANT: Fix the decimal point issue - divide by 100 to get proper percentage
                min_value = min(returns[current_drawdown_start_idx:i])
                drawdown_depth = (running_peak - min_value) / (1 + running_peak / 100)

                # Add this drawdown period to our list
                drawdown_periods.append(
                    {
                        "start_idx": current_drawdown_start_idx,
                        "end_idx": i,
                        "start_date": dates[current_drawdown_start_idx],
                        "end_date": dates[i],
                        "duration": i - current_drawdown_start_idx,
                        "max_drawdown": drawdown_depth,
                        # Calculate calendar days (not just trading days)
                        "calendar_days": (
                            date_objects[i] - date_objects[current_drawdown_start_idx]
                        ).days,
                    }
                )

                # Reset drawdown tracking
                in_drawdown = False
                current_drawdown_start_idx = None
                current_drawdown_start_value = None

        # Calculate current drawdown from peak - Fix decimal issue
        current_drawdown = (running_peak - value) / (1 + running_peak / 100)
        drawdowns.append(current_drawdown)  # Convert to percentage

        # Update maximum drawdown if this is a new max
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
            max_drawdown_idx = i

        # Detect the start of a new drawdown
        if current_drawdown > 0 and not in_drawdown:
            in_drawdown = True
            current_drawdown_start_idx = i
            current_drawdown_start_value = (
                running_peak  # Store the peak value, not the current value
            )

    # If we're still in a drawdown at the end, add that period
    if in_drawdown:
        # Calculate the depth (percentage) of this final drawdown period
        min_value = min(returns[current_drawdown_start_idx:])
        drawdown_depth = (running_peak - min_value) / (1 + running_peak / 100)

        # Add this drawdown period to our list
        drawdown_periods.append(
            {
                "start_idx": current_drawdown_start_idx,
                "end_idx": len(returns) - 1,
                "start_date": dates[current_drawdown_start_idx],
                "end_date": dates[-1],
                "duration": len(returns) - current_drawdown_start_idx,
                "max_drawdown": drawdown_depth,  # Convert to percentage
                # Calculate calendar days (not just trading days)
                "calendar_days": (
                    date_objects[-1] - date_objects[current_drawdown_start_idx]
                ).days,
            }
        )

    # FIX: Recalculate max_drawdown from the drawdowns list to ensure consistency
    if drawdowns:
        max_drawdown = max(drawdowns)
        max_drawdown_idx = drawdowns.index(max_drawdown)

    # FIX: Recalculate drawdown periods to ensure they have the correct max_drawdown value
    for i, period in enumerate(drawdown_periods):
        start_idx = period["start_idx"]
        end_idx = period["end_idx"]
        period_drawdowns = drawdowns[start_idx : end_idx + 1]
        if period_drawdowns:
            period_max_dd = max(period_drawdowns)
            drawdown_periods[i]["max_drawdown"] = period_max_dd

    # FIX: Sort drawdown periods by max_drawdown (descending) for proper ranking
    drawdown_periods.sort(key=lambda x: x["max_drawdown"], reverse=True)

    total_days_in_drawdown = sum(period["duration"] for period in drawdown_periods)
    # Use calendar days rather than trading days for significant periods
    significant_drawdown_days = sum(
        period["calendar_days"]
        for period in drawdown_periods
        if period["calendar_days"] > 20
    )

    # Find significant drawdown periods (duration > 20 calendar days)
    significant_periods = [p for p in drawdown_periods if p["calendar_days"] > 20]
    # Sort by max_drawdown (descending)
    significant_periods.sort(key=lambda x: x["max_drawdown"], reverse=True)

    # Print top 5 significant drawdown periods with actual dates
    print("\nTop 5 Significant Drawdown Periods (>20 calendar days):")
    print(
        f"{'Rank':<5} {'Trading Days':<12} {'Calendar Days':<14} {'Max Drawdown':<15} {'Start Date':<12} {'End Date':<12}"
    )
    print("-" * 70)

    for i, period in enumerate(significant_periods[:5], 1):
        print(
            f"{i:<5} {period['duration']:<12} {period['calendar_days']:<14} {period['max_drawdown']:.2f}%{' ':<9} {period['start_date']:<12} {period['end_date']:<12}"
        )

    if not significant_periods:
        print("No significant drawdown periods (>20 calendar days) found.")

    # Calculate average statistics
    avg_drawdown_length = (
        total_days_in_drawdown / len(drawdown_periods) if drawdown_periods else 0
    )
    # Calculate average calendar days
    avg_calendar_days = (
        sum(period["calendar_days"] for period in drawdown_periods)
        / len(drawdown_periods)
        if drawdown_periods
        else 0
    )
    non_zero_drawdowns = [d for d in drawdowns if d > 0]
    avg_drawdown_depth = (
        sum(non_zero_drawdowns) / len(non_zero_drawdowns) if non_zero_drawdowns else 0
    )

    # Debug info
    print(f"\nDrawdown Calculation Debug:")
    print(f"Overall Max Drawdown: {max_drawdown:.2f}%")
    print(f"Number of drawdown periods found: {len(drawdown_periods)}")
    if drawdown_periods:
        max_period = max(drawdown_periods, key=lambda x: x["max_drawdown"])
        print(
            f"Largest period drawdown: {max_period['max_drawdown']:.2f}% (Trading Days: {max_period['duration']}, Calendar Days: {max_period['calendar_days']}, {max_period['start_date']} to {max_period['end_date']})"
        )

    # Create the plot
    fig = plt.figure(figsize=(15, 12))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 1, 1], hspace=0.4, wspace=0.3)

    # Create underwater plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(range(len(drawdowns)), drawdowns, 0, color="red", alpha=0.3)
    ax1.plot(range(len(drawdowns)), drawdowns, color="red", linewidth=1)

    # Customize underwater plot
    ax1.set_title(f"Drawdown Over Time - {period_length} days")
    ax1.set_ylabel("Drawdown (%)")
    ax1.set_xlabel("Trading Days")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.1f}%".format(y)))

    # Add maximum drawdown line and annotation
    ax1.axhline(y=max_drawdown, color="darkred", linestyle="--", alpha=0.5)
    ax1.annotate(
        f"Maximum Drawdown: {max_drawdown:.2f}%\n"
        f"Average Drawdown: {avg_drawdown_depth:.2f}%\n"
        f"Avg Trading Days: {avg_drawdown_length:.1f}\n"
        f"Avg Calendar Days: {avg_calendar_days:.1f}",
        xy=(max_drawdown_idx, max_drawdown),
        xytext=(10, 10),
        textcoords="offset points",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.8),
    )

    # Create cumulative returns plot (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(range(len(returns)), returns, color="blue", linewidth=1.5)

    # Mark drawdown periods
    for period in drawdown_periods:
        ax2.axvspan(
            period["start_idx"],
            period["end_idx"],
            alpha=0.2,
            color="red",
            label="_" * period["start_idx"],
        )  # Unique labels to avoid duplicates in legend

    # Customize cumulative returns plot
    ax2.set_title(f"Cumulative Return with Drawdown Periods - {period_length} days")
    ax2.set_ylabel("Cumulative Return (%)")
    ax2.set_xlabel("Trading Days")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Plot drawdown durations (middle row)
    ax3 = fig.add_subplot(gs[1, 0])

    durations = [period["duration"] for period in drawdown_periods]
    calendar_days = [period["calendar_days"] for period in drawdown_periods]
    max_drawdowns = [period["max_drawdown"] for period in drawdown_periods]

    if durations:
        # Create a grouped bar chart with both trading days and calendar days
        x = np.arange(len(durations))
        width = 0.35

        ax3.bar(
            x - width / 2,
            durations,
            width,
            color="blue",
            alpha=0.7,
            label="Trading Days",
        )
        ax3.bar(
            x + width / 2,
            calendar_days,
            width,
            color="green",
            alpha=0.7,
            label="Calendar Days",
        )

        # Add drawdown depth as text
        for i, (_, cal_days, dd) in enumerate(
            zip(durations, calendar_days, max_drawdowns)
        ):
            ax3.text(
                i,
                cal_days + 1,
                f"{dd:.1f}%",
                ha="center",
                va="bottom",
                color="black",
                fontsize=8,
            )

        # Customize drawdown durations plot
        ax3.set_title("Drawdown Duration by Episode")
        ax3.set_ylabel("Duration (Days)")
        ax3.set_xlabel("Drawdown Episode")
        ax3.grid(True, linestyle="--", alpha=0.7)
        ax3.legend()
    else:
        ax3.text(
            0.5, 0.5, "No drawdown periods found", ha="center", va="center", fontsize=12
        )

    # Plot drawdown distribution (middle right)
    ax4 = fig.add_subplot(gs[1, 1])

    if drawdowns:
        non_zero_drawdowns = [d for d in drawdowns if d > 0]
        if non_zero_drawdowns:
            sns.histplot(non_zero_drawdowns, bins=20, kde=True, ax=ax4, color="green")

            # Add vertical line for mean and maximum
            mean_dd = np.mean(non_zero_drawdowns)
            median_dd = np.median(non_zero_drawdowns)

            ax4.axvline(
                mean_dd, color="red", linestyle="--", label=f"Mean: {mean_dd:.2f}%"
            )
            ax4.axvline(
                median_dd,
                color="blue",
                linestyle="--",
                label=f"Median: {median_dd:.2f}%",
            )
            ax4.axvline(
                max_drawdown,
                color="black",
                linestyle="-",
                label=f"Max: {max_drawdown:.2f}%",
            )

            # Customize drawdown distribution plot
            ax4.set_title("Drawdown Magnitude Distribution")
            ax4.set_xlabel("Drawdown (%)")
            ax4.set_ylabel("Frequency")
            ax4.legend()
            ax4.grid(True, linestyle="--", alpha=0.7)
        else:
            ax4.text(
                0.5,
                0.5,
                "No non-zero drawdowns found",
                ha="center",
                va="center",
                fontsize=12,
            )
    else:
        ax4.text(0.5, 0.5, "No drawdowns found", ha="center", va="center", fontsize=12)

    # Plot drawdown duration distribution (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])

    if durations:
        # Plot both trading days and calendar days as separate histograms
        sns.histplot(
            durations,
            bins=min(20, len(durations)),
            kde=True,
            ax=ax5,
            color="blue",
            alpha=0.4,
            label="Trading Days",
        )
        sns.histplot(
            calendar_days,
            bins=min(20, len(calendar_days)),
            kde=True,
            ax=ax5,
            color="green",
            alpha=0.4,
            label="Calendar Days",
        )

        # Add vertical line for mean and maximum of calendar days
        mean_duration = np.mean(durations)
        mean_calendar = np.mean(calendar_days)
        median_duration = np.median(durations)
        median_calendar = np.median(calendar_days)
        max_calendar = max(calendar_days)

        ax5.axvline(
            mean_calendar,
            color="darkgreen",
            linestyle="--",
            label=f"Mean Calendar: {mean_calendar:.1f} days",
        )
        ax5.axvline(
            median_calendar,
            color="green",
            linestyle=":",
            label=f"Median Calendar: {median_calendar:.1f} days",
        )
        ax5.axvline(
            max_calendar,
            color="green",
            linestyle="-",
            label=f"Max Calendar: {max_calendar:.0f} days",
        )

        # Customize drawdown duration distribution plot
        ax5.set_title("Drawdown Duration Distribution")
        ax5.set_xlabel("Duration (Days)")
        ax5.set_ylabel("Frequency")
        ax5.legend()
        ax5.grid(True, linestyle="--", alpha=0.7)
    else:
        ax5.text(
            0.5, 0.5, "No drawdown periods found", ha="center", va="center", fontsize=12
        )

    # Plot drawdown scatter (bottom right)
    ax6 = fig.add_subplot(gs[2, 1])

    if durations and max_drawdowns:
        # Scatter plot with calendar days instead of trading days
        ax6.scatter(
            max_drawdowns,
            calendar_days,
            alpha=0.7,
            c="green",
            s=50,
            label="Calendar Days",
        )
        ax6.scatter(
            max_drawdowns, durations, alpha=0.7, c="blue", s=30, label="Trading Days"
        )

        # Add regression line for calendar days
        if len(calendar_days) > 1:
            z = np.polyfit(max_drawdowns, calendar_days, 1)
            p = np.poly1d(z)
            x_range = np.linspace(min(max_drawdowns), max(max_drawdowns), 100)
            ax6.plot(x_range, p(x_range), "g--", alpha=0.7)

            # Calculate correlation
            corr = np.corrcoef(max_drawdowns, calendar_days)[0, 1]
            ax6.text(
                0.05,
                0.95,
                f"Calendar Day Correlation: {corr:.2f}",
                transform=ax6.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
            )

        # Customize scatter plot
        ax6.set_title("Drawdown Magnitude vs Duration")
        ax6.set_xlabel("Maximum Drawdown (%)")
        ax6.set_ylabel("Duration (Days)")
        ax6.grid(True, linestyle="--", alpha=0.7)
        ax6.legend()
    else:
        ax6.text(
            0.5, 0.5, "No drawdown periods found", ha="center", va="center", fontsize=12
        )

    # Add overall title
    plt.suptitle(
        f"Drawdown Analysis ({test_start_date} to {test_end_date})", fontsize=16, y=0.99
    )

    # FIX: Replace tight_layout with more manual control of the layout
    # First adjust all subplots
    fig.subplots_adjust(
        left=0.1, right=0.9, bottom=0.1, top=0.9, hspace=0.4, wspace=0.3
    )

    # Then manually adjust for the suptitle
    fig.subplots_adjust(top=0.92)  # Reserve space for suptitle

    save_path = os.path.join(
        output_dir, f"{portfolio_name}_drawdown_analysis_{period_length}d.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Return drawdown statistics with date information
    return {
        "max_drawdown": max_drawdown,
        "avg_drawdown": avg_drawdown_depth,
        "total_drawdown_days": total_days_in_drawdown,
        "significant_drawdown_days": significant_drawdown_days,
        "avg_drawdown_length": avg_drawdown_length,
        "avg_calendar_days": avg_calendar_days,
        "drawdown_periods": len(drawdown_periods),
        "significant_periods": len(significant_periods),
        "max_drawdown_duration": (
            max(period["duration"] for period in drawdown_periods)
            if drawdown_periods
            else 0
        ),
        "max_calendar_duration": (
            max(period["calendar_days"] for period in drawdown_periods)
            if drawdown_periods
            else 0
        ),
        "drawdown_durations": [period["duration"] for period in drawdown_periods],
        "calendar_durations": [period["calendar_days"] for period in drawdown_periods],
        "drawdown_magnitudes": [period["max_drawdown"] for period in drawdown_periods],
        "top_significant_periods": significant_periods[:5],
    }


def plot_drawdown_distributions(
    simulation_results,
    actual_max_drawdown,
    actual_dd_duration,
    period_length,
    output_dir,
    portfolio_name,
):
    """
    Plot side-by-side distributions of drawdown magnitude and duration
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot drawdown magnitude distribution
    max_drawdowns = simulation_results["max_drawdowns"]

    # max_drawdowns values are approximately percent-scale (same formula as analyze_drawdowns).
    # Both use: (peak - value) / (1 + peak / 100) on percent-scale inputs. No unit conversion needed.
    sns.histplot(max_drawdowns, kde=True, bins=30, ax=ax1, color="blue", alpha=0.6)
    ax1.axvline(
        x=actual_max_drawdown,
        color="r",
        linestyle="--",
        label=f"Actual: {actual_max_drawdown:.2f}%",
    )

    # Add percentile information
    dd_percentile = stats.percentileofscore(max_drawdowns, actual_max_drawdown)
    ax1.text(
        0.05,
        0.95,
        f"Actual Percentile: {dd_percentile:.1f}%",
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Calculate statistics
    dd_mean = np.mean(max_drawdowns)
    dd_median = np.median(max_drawdowns)
    dd_std = np.std(max_drawdowns)
    dd_5th = np.percentile(max_drawdowns, 5)
    dd_95th = np.percentile(max_drawdowns, 95)

    # Add statistics
    stats_text = (
        f"Mean: {dd_mean:.2f}%\n"
        f"Median: {dd_median:.2f}%\n"
        f"Std Dev: {dd_std:.2f}%\n"
        f"5th %ile: {dd_5th:.2f}%\n"
        f"95th %ile: {dd_95th:.2f}%"
    )

    ax1.text(
        0.95,
        0.95,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Customize plot
    ax1.set_title(
        f"Maximum Drawdown Distribution - {period_length} Day Forward Test", fontsize=14
    )
    ax1.set_xlabel("Maximum Drawdown (%)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot drawdown duration distribution
    max_dd_durations = simulation_results["max_drawdown_durations"]

    # For the simulated durations, estimate calendar days
    # Typically trading days are ~252/365 = 0.69 of calendar days, so we'll scale by 1.45
    estimated_calendar_durations = [d * 1.45 for d in max_dd_durations]

    sns.histplot(
        estimated_calendar_durations,
        kde=True,
        bins=30,
        ax=ax2,
        color="green",
        alpha=0.6,
        label="Estimated Calendar Days",
    )
    ax2.axvline(
        x=actual_dd_duration,
        color="r",
        linestyle="--",
        label=f"Actual: {actual_dd_duration} calendar days",
    )

    # Add percentile information
    duration_percentile = stats.percentileofscore(
        estimated_calendar_durations, actual_dd_duration
    )
    ax2.text(
        0.05,
        0.95,
        f"Actual Percentile: {duration_percentile:.1f}%",
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Calculate statistics
    dur_mean = np.mean(estimated_calendar_durations)
    dur_median = np.median(estimated_calendar_durations)
    dur_std = np.std(estimated_calendar_durations)
    dur_5th = np.percentile(estimated_calendar_durations, 5)
    dur_95th = np.percentile(estimated_calendar_durations, 95)

    # Add statistics
    stats_text = (
        f"Mean: {dur_mean:.1f} days\n"
        f"Median: {dur_median:.1f} days\n"
        f"Std Dev: {dur_std:.1f} days\n"
        f"5th %ile: {dur_5th:.1f} days\n"
        f"95th %ile: {dur_95th:.1f} days"
    )

    ax2.text(
        0.95,
        0.95,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Customize plot
    ax2.set_title(
        f"Maximum Drawdown Duration Distribution - {period_length} Day Forward Test",
        fontsize=14,
    )
    ax2.set_xlabel("Maximum Drawdown Duration (Calendar Days)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Save the figure
    plt.tight_layout()
    save_path = os.path.join(
        output_dir, f"{portfolio_name}_drawdown_distributions_{period_length}d.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Return statistics
    return {
        "dd_mean": dd_mean,
        "dd_median": dd_median,
        "dd_std": dd_std,
        "dd_5th": dd_5th,
        "dd_95th": dd_95th,
        "dd_percentile": dd_percentile,
        "dur_mean": dur_mean,
        "dur_median": dur_median,
        "dur_std": dur_std,
        "dur_5th": dur_5th,
        "dur_95th": dur_95th,
        "dur_percentile": duration_percentile,
    }


def run_walk_forward_test(
    dates, returns, test_period_length, output_dir, portfolio_name
):
    """
    Run a walk-forward test where we use data up to a certain point to predict forward
    and compare with actual returns
    """
    if len(returns) <= test_period_length:
        print(f"Not enough data for walk-forward test of {test_period_length} days")
        return None

    # Split data into training and test sets
    train_returns = returns[:-test_period_length]
    test_returns = returns[-test_period_length:]
    test_dates = dates[-test_period_length:]

    # Only run simulation if we have enough training data
    if len(train_returns) < 30:  # Require at least 30 days for training
        print(
            f"Not enough training data for walk-forward test of {test_period_length} days"
        )
        return None

    # Run simulation on training data
    print(f"\n--- Running Walk-Forward Test for {test_period_length} days ---")
    print(
        f"Training on {len(train_returns)} days, testing on {test_period_length} days"
    )

    num_simulations = 10000
    simulation_results = run_monte_carlo_simulation(
        train_returns, num_simulations, test_period_length, annual_periods=252
    )

    # Calculate actual cumulative return path
    actual_returns = [0.0]  # Start with 0% return
    cumulative_return = 0.0

    for r in test_returns:
        # Convert daily percentage return to decimal
        r_decimal = r / 100.0

        # Calculate new cumulative return (compounded)
        cumulative_return = (1 + cumulative_return / 100) * (1 + r_decimal) * 100 - 100
        actual_returns.append(cumulative_return)

    # Calculate actual metrics
    actual_final_return = actual_returns[-1]

    # Calculate actual drawdown statistics using our helper function
    test_start_date = dates[-test_period_length]
    test_end_date = dates[-1]

    drawdown_stats = analyze_drawdowns(
        actual_returns,
        output_dir,
        test_period_length,
        test_start_date,
        test_end_date,
        portfolio_name,
        dates=[test_dates[0]]
        + test_dates,  # Add an initial date for the 0% return point
    )

    actual_max_drawdown = drawdown_stats["max_drawdown"]
    actual_max_dd_duration = drawdown_stats["max_drawdown_duration"]

    # Plot drawdown distributions for simulations vs actual with portfolio name added
    if test_period_length >= 63:  # Only for periods of 3+ months
        dd_distribution_stats = plot_drawdown_distributions(
            simulation_results,
            actual_max_drawdown,
            actual_max_dd_duration,
            test_period_length,
            output_dir,
            portfolio_name,
        )

    # Calculate annualized metrics if appropriate
    if test_period_length >= 20:  # Only calculate for meaningful periods
        actual_years = test_period_length / 252
        actual_annualized_return = (
            (1 + actual_final_return / 100) ** (1 / actual_years) - 1
        ) * 100

        # Calculate actual Sharpe ratio
        actual_volatility = np.std(test_returns) * np.sqrt(252)
        actual_sharpe = (
            (actual_annualized_return / actual_volatility)
            if actual_volatility != 0
            else 0
        )
    else:
        actual_annualized_return = (
            actual_final_return  # For very short periods, use simple return
        )
        actual_sharpe = 0

    # Get percentile rank of actual result within simulation
    final_returns = simulation_results["final_returns"]
    actual_percentile = stats.percentileofscore(final_returns, actual_final_return)

    # Plot simulation with actual path overlaid
    plt.figure(figsize=(12, 8))

    # Plot percentile bands
    percentiles = simulation_results["percentiles"]
    x = range(len(percentiles["50"]))
    plt.fill_between(
        x,
        percentiles["5"],
        percentiles["95"],
        color="lightblue",
        alpha=0.3,
        label="5th-95th Percentile",
    )
    plt.fill_between(
        x,
        percentiles["25"],
        percentiles["75"],
        color="blue",
        alpha=0.3,
        label="25th-75th Percentile",
    )
    plt.plot(x, percentiles["50"], "b-", linewidth=2, label="Median Path")

    # Find the best and worst paths based on final return
    all_paths = simulation_results["paths"]
    best_path_idx = np.argmax(final_returns)
    worst_path_idx = np.argmin(final_returns)

    # Plot actual path
    plt.plot(
        x,
        actual_returns,
        "orange",
        linewidth=3,
        label=f"Actual ({actual_final_return:.2f}%, {actual_percentile:.1f}%ile)",
    )

    # Format plot
    if test_period_length <= 63:  # 3 months
        period_desc = f"{test_period_length} days (~3 months)"
    elif test_period_length <= 126:  # 6 months
        period_desc = f"{test_period_length} days (~6 months)"
    elif test_period_length <= 252:  # 1 year
        period_desc = f"{test_period_length} days (~1 year)"
    elif test_period_length <= 504:  # 2 years
        period_desc = f"{test_period_length} days (~2 years)"
    else:
        period_desc = f"{test_period_length} days"

    plt.title(
        f"Walk-Forward Test: {period_desc} ({test_start_date} to {test_end_date})",
        fontsize=14,
    )
    plt.xlabel("Trading Days", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")

    # Save the figure
    save_path = os.path.join(
        output_dir, f"{portfolio_name}_walk_forward_{test_period_length}d.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.close()

    # Generate CAGR distribution plot for longer periods
    if test_period_length >= 252:  # Only for 1-year or longer periods
        # Calculate CAGR for all simulations
        years = test_period_length / 252
        cagr_values = [
            ((1 + ret / 100) ** (1 / years) - 1) * 100 for ret in final_returns
        ]
        actual_cagr = ((1 + actual_final_return / 100) ** (1 / years) - 1) * 100

        plt.figure(figsize=(10, 6))
        sns.histplot(cagr_values, kde=True, bins=50)
        plt.axvline(
            x=actual_cagr,
            color="r",
            linestyle="--",
            label=f"Actual CAGR: {actual_cagr:.2f}%",
        )

        # Add percentile information
        cagr_percentile = stats.percentileofscore(cagr_values, actual_cagr)
        plt.text(
            0.05,
            0.95,
            f"Actual CAGR Percentile: {cagr_percentile:.1f}%",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        plt.title(f"CAGR Distribution - {period_desc} Forward Test", fontsize=14)
        plt.xlabel("CAGR (%)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Save the CAGR distribution plot
        cagr_plot_path = os.path.join(
            output_dir, f"{portfolio_name}_cagr_distribution_{test_period_length}d.png"
        )
        plt.savefig(cagr_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        # Calculate statistics for CAGR distribution
        cagr_mean = np.mean(cagr_values)
        cagr_median = np.median(cagr_values)
        cagr_std = np.std(cagr_values)
        cagr_5th = np.percentile(cagr_values, 5)
        cagr_95th = np.percentile(cagr_values, 95)

        print(f"\nCAGR Distribution Statistics:")
        print(f"Mean CAGR: {cagr_mean:.2f}%")
        print(f"Median CAGR: {cagr_median:.2f}%")
        print(f"Standard Deviation: {cagr_std:.2f}%")
        print(f"5th Percentile: {cagr_5th:.2f}%")
        print(f"95th Percentile: {cagr_95th:.2f}%")
        print(f"Actual CAGR: {actual_cagr:.2f}% (Percentile: {cagr_percentile:.1f}%)")

    # Print summary statistics
    print(f"\nWalk-Forward Test Results for {period_desc}:")
    print(f"Test Period: {test_start_date} to {test_end_date}")
    print(f"Actual Cumulative Return: {actual_final_return:.2f}%")

    if test_period_length >= 20:
        print(f"Actual Annualized Return: {actual_annualized_return:.2f}%")
        print(f"Actual Sharpe Ratio: {actual_sharpe:.2f}")
        print(f"Actual Max Drawdown: {actual_max_drawdown:.2f}%")

    print(f"Percentile Rank: {actual_percentile:.1f}%")

    # Calculate forecast accuracy metrics
    median_return = percentiles["50"][-1]
    error = actual_final_return - median_return
    percent_error = (
        (error / abs(median_return)) * 100 if abs(median_return) > 0.01 else 0
    )

    print(f"Median Forecast: {median_return:.2f}%")
    print(f"Forecast Error: {error:.2f}% ({percent_error:.2f}%)")

    # Check if actual was within confidence intervals
    in_90_interval = (
        percentiles["5"][-1] <= actual_final_return <= percentiles["95"][-1]
    )
    in_50_interval = (
        percentiles["25"][-1] <= actual_final_return <= percentiles["75"][-1]
    )

    print(f"Actual within 90% Confidence Interval: {in_90_interval}")
    print(f"Actual within 50% Confidence Interval: {in_50_interval}")

    # Print drawdown statistics
    print(f"\nDrawdown Analysis:")
    print(f"Maximum Drawdown: {actual_max_drawdown:.2f}%")
    print(f"Average Drawdown: {drawdown_stats['avg_drawdown']:.2f}%")
    print(
        f"Total Days in Drawdown: {drawdown_stats['total_drawdown_days']} trading days"
    )
    print(f"Number of Drawdown Periods: {drawdown_stats['drawdown_periods']}")
    print(
        f"Average Trading Day Length: {drawdown_stats['avg_drawdown_length']:.1f} days"
    )
    print(
        f"Average Calendar Day Length: {drawdown_stats['avg_calendar_days']:.1f} days"
    )
    print(
        f"Maximum Drawdown Duration: {drawdown_stats['max_drawdown_duration']} trading days, {drawdown_stats['max_calendar_duration']} calendar days"
    )

    # Results dictionary with both trading and calendar days
    result = {
        "period_length": test_period_length,
        "test_start_date": test_start_date,
        "test_end_date": test_end_date,
        "actual_final_return": actual_final_return,
        "actual_annualized_return": (
            actual_annualized_return if test_period_length >= 20 else None
        ),
        "actual_sharpe": actual_sharpe if test_period_length >= 20 else None,
        "actual_max_drawdown": actual_max_drawdown,
        "actual_dd_duration_trading": drawdown_stats["max_drawdown_duration"],
        "actual_dd_duration_calendar": actual_max_dd_duration,
        "actual_percentile": actual_percentile,
        "median_forecast": median_return,
        "forecast_error": error,
        "percent_error": percent_error,
        "in_90_interval": in_90_interval,
        "in_50_interval": in_50_interval,
        # Add drawdown statistics
        "avg_drawdown": drawdown_stats["avg_drawdown"],
        "total_drawdown_days": drawdown_stats["total_drawdown_days"],
        "drawdown_periods": drawdown_stats["drawdown_periods"],
        "avg_drawdown_length_trading": drawdown_stats["avg_drawdown_length"],
        "avg_drawdown_length_calendar": drawdown_stats["avg_calendar_days"],
    }

    # Add CAGR and drawdown distribution statistics for longer periods
    if test_period_length >= 252:
        result.update(
            {
                "cagr_mean": cagr_mean,
                "cagr_median": cagr_median,
                "cagr_std": cagr_std,
                "cagr_5th": cagr_5th,
                "cagr_95th": cagr_95th,
                "actual_cagr": actual_cagr,
                "cagr_percentile": cagr_percentile,
            }
        )

    # Add drawdown distribution statistics for periods >= 3 months
    if test_period_length >= 63:
        result.update(
            {
                "dd_mean": dd_distribution_stats["dd_mean"],
                "dd_median": dd_distribution_stats["dd_median"],
                "dd_std": dd_distribution_stats["dd_std"],
                "dd_5th": dd_distribution_stats["dd_5th"],
                "dd_95th": dd_distribution_stats["dd_95th"],
                "dd_percentile": dd_distribution_stats["dd_percentile"],
                "dur_mean": dd_distribution_stats["dur_mean"],
                "dur_median": dd_distribution_stats["dur_median"],
                "dur_std": dd_distribution_stats["dur_std"],
                "dur_5th": dd_distribution_stats["dur_5th"],
                "dur_95th": dd_distribution_stats["dur_95th"],
                "dur_percentile": dd_distribution_stats["dur_percentile"],
            }
        )

    return result


def run_rolling_walk_forward_test(
    dates,
    returns,
    train_period_length,
    test_period_length,
    output_dir,
    portfolio_name,
    step_size=None,
):
    """
    Run multiple walk-forward tests by rolling through the historical data

    Parameters:
    -----------
    dates : list
        List of date strings for the full dataset
    returns : list
        List of daily returns for the full dataset
    train_period_length : int
        Length of the training period in trading days
    test_period_length : int
        Length of the test period in trading days
    output_dir : str
        Directory to save output files
    portfolio_name : str
        Name of the portfolio for file naming
    step_size : int, optional
        Number of days to step forward for each iteration (defaults to test_period_length)
    """
    if len(returns) < (train_period_length + test_period_length):
        print(
            f"Not enough data for rolling walk-forward test. Need at least {train_period_length + test_period_length} days."
        )
        return None

    # If step_size is not specified, default to test_period_length (non-overlapping periods)
    if step_size is None:
        step_size = test_period_length

    # Create a specific directory for rolling walk results
    rolling_dir = os.path.join(output_dir, f"{portfolio_name}_rolling_walk")
    os.makedirs(rolling_dir, exist_ok=True)

    # Determine how many iterations we can run
    available_test_days = len(returns) - train_period_length
    num_iterations = max(1, available_test_days // step_size)

    print(f"\n--- Running Rolling Walk-Forward Test ---")
    print(f"Training period: {train_period_length} days")
    print(f"Test period: {test_period_length} days")
    print(f"Step size: {step_size} days")
    print(f"Number of iterations: {num_iterations}")

    # Lists to store results
    period_labels = []
    actual_returns = []
    forecast_returns = []
    actual_cagrs = []
    forecast_cagrs = []
    max_drawdowns = []
    actual_percentiles = []

    # Run iterations
    for i in range(num_iterations):
        start_idx = i * step_size
        train_end_idx = start_idx + train_period_length
        test_end_idx = min(train_end_idx + test_period_length, len(returns))

        # Check if we have enough data for this test period
        if test_end_idx - train_end_idx < test_period_length:
            print(
                f"Skipping iteration {i + 1} - not enough data for complete test period"
            )
            continue

        # Extract data for this iteration
        train_data = returns[start_idx:train_end_idx]
        test_data = returns[train_end_idx:test_end_idx]
        test_dates = dates[train_end_idx:test_end_idx]

        train_start_date = dates[start_idx]
        train_end_date = dates[train_end_idx - 1]
        test_start_date = dates[train_end_idx]
        test_end_date = dates[test_end_idx - 1]

        period_label = f"{test_start_date} to {test_end_date}"
        period_labels.append(period_label)

        print(f"\nIteration {i + 1} of {num_iterations}:")
        print(
            f"Training period: {train_start_date} to {train_end_date} ({len(train_data)} days)"
        )
        print(
            f"Test period: {test_start_date} to {test_end_date} ({len(test_data)} days)"
        )

        # Run Monte Carlo simulation on the training data
        num_simulations = 10000
        simulation_results = run_monte_carlo_simulation(
            train_data, num_simulations, len(test_data), annual_periods=252
        )

        # Calculate actual cumulative return path
        actual_return_path = [0.0]  # Start with 0% return
        cumulative_return = 0.0

        for r in test_data:
            # Convert daily percentage return to decimal
            r_decimal = r / 100.0

            # Calculate new cumulative return (compounded)
            cumulative_return = (1 + cumulative_return / 100) * (
                1 + r_decimal
            ) * 100 - 100
            actual_return_path.append(cumulative_return)

        actual_final_return = actual_return_path[-1]
        actual_returns.append(actual_final_return)

        # Get forecast (median) from simulation
        median_return = simulation_results["percentiles"]["50"][-1]
        forecast_returns.append(median_return)

        # Calculate actual vs forecasted CAGR for meaningful periods
        if test_period_length >= 20:
            actual_years = len(test_data) / 252
            actual_annualized_return = (
                (1 + actual_final_return / 100) ** (1 / actual_years) - 1
            ) * 100
            forecast_annualized_return = (
                (1 + median_return / 100) ** (1 / actual_years) - 1
            ) * 100

            actual_cagrs.append(actual_annualized_return)
            forecast_cagrs.append(forecast_annualized_return)

        # Calculate actual drawdown statistics using our helper function
        drawdown_stats = analyze_drawdowns(
            actual_return_path,
            rolling_dir,
            len(test_data),
            test_start_date,
            test_end_date,
            f"{portfolio_name}_iter{i + 1}",
            dates=[test_dates[0]]
            + test_dates,  # Add an initial date for the 0% return point
        )

        max_drawdowns.append(drawdown_stats["max_drawdown"])

        # Get percentile rank of actual result within simulation
        final_returns = simulation_results["final_returns"]
        percentile = stats.percentileofscore(final_returns, actual_final_return)
        actual_percentiles.append(percentile)

        # Save the Monte Carlo simulation plot with actual path
        plt.figure(figsize=(12, 8))

        # Plot percentile bands
        percentiles = simulation_results["percentiles"]
        x = range(len(percentiles["50"]))
        plt.fill_between(
            x,
            percentiles["5"],
            percentiles["95"],
            color="lightblue",
            alpha=0.3,
            label="5th-95th Percentile",
        )
        plt.fill_between(
            x,
            percentiles["25"],
            percentiles["75"],
            color="blue",
            alpha=0.3,
            label="25th-75th Percentile",
        )
        plt.plot(x, percentiles["50"], "b-", linewidth=2, label="Median Path")

        # Plot actual path
        plt.plot(
            x,
            actual_return_path,
            "orange",
            linewidth=3,
            label=f"Actual ({actual_final_return:.2f}%, {percentile:.1f}%ile)",
        )

        # Format plot
        plt.title(
            f"Rolling Walk-Forward Test: Iteration {i + 1} ({test_start_date} to {test_end_date})",
            fontsize=14,
        )
        plt.xlabel("Trading Days", fontsize=12)
        plt.ylabel("Cumulative Return (%)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best")

        # Save the figure
        save_path = os.path.join(
            rolling_dir, f"{portfolio_name}_rolling_iter{i + 1}.png"
        )
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.tight_layout()
        plt.close()

        print(
            f"Actual Return: {actual_final_return:.2f}%, Forecast: {median_return:.2f}%"
        )
        print(
            f"Percentile: {percentile:.1f}%, Max Drawdown: {drawdown_stats['max_drawdown']:.2f}%"
        )

        if len(actual_cagrs) > 0:
            print(
                f"Actual CAGR: {actual_cagrs[-1]:.2f}%, Forecast CAGR: {forecast_cagrs[-1]:.2f}%"
            )

    # Create summary visualizations
    if len(period_labels) > 0:
        # Create date labels for x-axis (convert to month/year format)
        date_labels = []
        for period in period_labels:
            start_date = period.split(" to ")[
                0
            ]  # Extract start date from "YYYY-MM-DD to YYYY-MM-DD"
            try:
                dt = datetime.strptime(start_date, "%Y-%m-%d")
                date_labels.append(dt.strftime("%b %Y"))  # Format as "Jan 2023"
            except:
                # Fallback to iteration number if date conversion fails
                date_labels.append(f"Period {len(date_labels) + 1}")

        # Create a plot comparing actual vs forecast returns
        plt.figure(figsize=(14, 7))

        # Set up index for bars
        indices = np.arange(len(period_labels))
        width = 0.35

        # Create bar chart of actual vs forecasted returns
        plt.bar(
            indices - width / 2,
            actual_returns,
            width,
            label="Actual Return",
            color="green",
            alpha=0.7,
        )
        plt.bar(
            indices + width / 2,
            forecast_returns,
            width,
            label="Forecast Return",
            color="blue",
            alpha=0.7,
        )

        # Add value labels on bars
        for i, v in enumerate(actual_returns):
            plt.text(
                i - width / 2,
                v + 1,
                f"{v:.1f}%",
                ha="center",
                fontsize=9,
                rotation=90 if abs(v) > 20 else 0,
            )

        for i, v in enumerate(forecast_returns):
            plt.text(
                i + width / 2,
                v + 1,
                f"{v:.1f}%",
                ha="center",
                fontsize=9,
                rotation=90 if abs(v) > 20 else 0,
            )

        # Customize plot
        plt.xlabel("Test Period Start")
        plt.ylabel("Cumulative Return (%)")
        plt.title(
            f"Rolling Walk-Forward Test: Actual vs Forecast Returns - {portfolio_name}"
        )
        plt.xticks(indices, date_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis="y")
        plt.legend()

        # Add percentile labels
        for i, pct in enumerate(actual_percentiles):
            plt.text(
                i,
                max(actual_returns[i], forecast_returns[i]) + 5,
                f"{pct:.0f}%ile",
                ha="center",
                fontsize=9,
                color="red",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(
                rolling_dir, f"{portfolio_name}_rolling_returns_comparison.png"
            ),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create CAGR comparison chart if we have CAGR data
        if len(actual_cagrs) > 0:
            plt.figure(figsize=(14, 7))

            # Create bar chart of actual vs forecasted CAGR
            plt.bar(
                indices - width / 2,
                actual_cagrs,
                width,
                label="Actual CAGR",
                color="green",
                alpha=0.7,
            )
            plt.bar(
                indices + width / 2,
                forecast_cagrs,
                width,
                label="Forecast CAGR",
                color="blue",
                alpha=0.7,
            )

            # Add value labels on bars
            for i, v in enumerate(actual_cagrs):
                plt.text(
                    i - width / 2,
                    v + 1,
                    f"{v:.1f}%",
                    ha="center",
                    fontsize=9,
                    rotation=90 if abs(v) > 20 else 0,
                )

            for i, v in enumerate(forecast_cagrs):
                plt.text(
                    i + width / 2,
                    v + 1,
                    f"{v:.1f}%",
                    ha="center",
                    fontsize=9,
                    rotation=90 if abs(v) > 20 else 0,
                )

            # Customize plot
            plt.xlabel("Test Period Start")
            plt.ylabel("Annualized Return (%)")
            plt.title(
                f"Rolling Walk-Forward Test: Actual vs Forecast CAGR - {portfolio_name}"
            )
            plt.xticks(indices, date_labels, rotation=45)
            plt.grid(True, alpha=0.3, axis="y")
            plt.legend()

            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    rolling_dir, f"{portfolio_name}_rolling_cagr_comparison.png"
                ),
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        # Create a plot of max drawdowns
        plt.figure(figsize=(14, 7))
        plt.bar(indices, max_drawdowns, color="red", alpha=0.7)

        # Add value labels on bars
        for i, v in enumerate(max_drawdowns):
            plt.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

        # Customize plot
        plt.xlabel("Test Period Start")
        plt.ylabel("Maximum Drawdown (%)")
        plt.title(f"Rolling Walk-Forward Test: Maximum Drawdowns - {portfolio_name}")
        plt.xticks(indices, date_labels, rotation=45)
        plt.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(
            os.path.join(rolling_dir, f"{portfolio_name}_rolling_drawdowns.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Create a DataFrame with all results
        results_data = {
            "Iteration": list(range(1, len(period_labels) + 1)),
            "Period": period_labels,
            "Actual_Return": actual_returns,
            "Forecast_Return": forecast_returns,
            "Error": [a - f for a, f in zip(actual_returns, forecast_returns)],
            "Percentile": actual_percentiles,
            "Max_Drawdown": max_drawdowns,
        }

        if len(actual_cagrs) > 0:
            results_data["Actual_CAGR"] = actual_cagrs
            results_data["Forecast_CAGR"] = forecast_cagrs
            results_data["CAGR_Error"] = [
                a - f for a, f in zip(actual_cagrs, forecast_cagrs)
            ]

        results_df = pd.DataFrame(results_data)

        # Save results to CSV
        csv_path = os.path.join(rolling_dir, f"{portfolio_name}_rolling_results.csv")
        results_df.to_csv(csv_path, index=False)

        # Print summary statistics
        print("\nRolling Walk-Forward Test Summary:")
        print(f"Average Actual Return: {np.mean(actual_returns):.2f}%")
        print(f"Average Forecast Return: {np.mean(forecast_returns):.2f}%")
        print(
            f"Average Error: {np.mean([a - f for a, f in zip(actual_returns, forecast_returns)]):.2f}%"
        )
        print(f"Average Percentile: {np.mean(actual_percentiles):.1f}%")
        print(f"Average Max Drawdown: {np.mean(max_drawdowns):.2f}%")

        if len(actual_cagrs) > 0:
            print(f"Average Actual CAGR: {np.mean(actual_cagrs):.2f}%")
            print(f"Average Forecast CAGR: {np.mean(forecast_cagrs):.2f}%")

        print(f"\nRolling walk-forward test results saved to: {rolling_dir}/")

        return results_df

    return None


def main():
    """
    Main function to run the Monte Carlo analysis on a Composer portfolio
    """
    # Default composer URL if none is provided
    default_url = "https://app.composer.trade/symphony/IxUYGLhjD2rF1Xi2GEmI/details"

    # Get input from user
    symphony_url = (
        input(f"Enter Composer Symphony URL (default: {default_url}): ") or default_url
    )

    # Set date range
    today = date.today().strftime("%Y-%m-%d")
    start_date = "2000-01-01"  # Fetch all available data
    end_date = today

    # Output directory
    output_dir = "composer_monte_carlo_results"
    os.makedirs(output_dir, exist_ok=True)

    # Fetch backtest data from Composer
    print(f"Fetching data from Composer: {symphony_url}")
    allocations_df, symphony_name, tickers = fetch_backtest(
        symphony_url, start_date, end_date
    )

    # Clean symphony name for file naming (remove special characters that could cause file system issues)
    clean_symphony_name = "".join(
        c if c.isalnum() or c in [" ", "_", "-"] else "_" for c in symphony_name
    )
    clean_symphony_name = clean_symphony_name.replace(" ", "_")

    # Calculate daily returns
    print(f"Calculating daily returns for {symphony_name}...")
    daily_returns, dates = calculate_portfolio_returns(allocations_df, tickers)

    # Convert dates to strings for easier handling
    date_strs = [d.strftime("%Y-%m-%d") for d in dates]

    # Check if we have enough data
    if len(daily_returns) < 60:  # Need at least 60 days of data
        print(
            "Error: Not enough historical data for Monte Carlo analysis (minimum 60 days required)."
        )
        return

    # Make sure date_strs and daily_returns have the same length
    if len(date_strs) != len(daily_returns):
        print(
            f"Warning: Length mismatch - date_strs: {len(date_strs)}, daily_returns: {len(daily_returns)}"
        )
        # Trim to the shorter length
        min_length = min(len(date_strs), len(daily_returns))
        date_strs = date_strs[:min_length]
        # For daily_returns, we need to handle it as a pandas Series
        if isinstance(daily_returns, pd.Series):
            daily_returns = daily_returns.iloc[:min_length]
        else:
            daily_returns = daily_returns[:min_length]

    # Create the DataFrame
    returns_df = pd.DataFrame({"Date": date_strs, "Daily_Return": daily_returns})

    # Define simulation parameters
    annual_periods = 252

    # Set simulation lengths for different time periods
    simulation_length_3m = int(annual_periods / 4)  # 3 months (1 quarter)
    simulation_length_6m = int(annual_periods / 2)  # 6 months
    simulation_length_1y = annual_periods  # 1 year
    simulation_length_2y = annual_periods * 2  # 2 years

    # Save the daily returns data before running the tests with symphony name prefixed
    returns_path = os.path.join(output_dir, f"{clean_symphony_name}_daily_returns.csv")
    returns_df.to_csv(returns_path, index=False)
    print(f"Daily returns saved to: {returns_path}")

    print(f"\nAnalyzing portfolio: {symphony_name}")
    print(f"Historical data period: {date_strs[0]} to {date_strs[-1]}")
    print(f"Total trading days: {len(daily_returns)}")

    # Ask user which test to run
    print("\nAvailable test modes:")
    print(
        "1. Standard Walk-Forward (train on all data except last N days, test on last N days)"
    )
    print(
        "2. Rolling Walk (start at beginning, train on x days, simulate y forward days)"
    )
    print("3. Both")

    mode = input("Enter test mode (1, 2, or 3, default: 1): ") or "1"

    # Run standard walk-forward tests if selected
    if mode in ["1", "3"]:
        print("\n------------------------------------------")
        print("Standard Walk-Forward Tests")
        print("------------------------------------------")

        # Define walk-forward test periods
        walk_forward_test_periods = [
            simulation_length_3m,  # ~3 months
            simulation_length_6m,  # ~6 months
            simulation_length_1y,  # ~1 year
        ]

        # Add 2-year test if we have enough data
        if len(daily_returns) >= (
            simulation_length_2y + 60
        ):  # Need at least 60 days of training data
            walk_forward_test_periods.append(simulation_length_2y)

        walk_forward_results = []

        for period_length in walk_forward_test_periods:
            if period_length <= len(daily_returns):
                try:
                    # Added symphony_name to the function call
                    result = run_walk_forward_test(
                        date_strs,
                        daily_returns.tolist(),
                        period_length,
                        output_dir,
                        clean_symphony_name,
                    )
                    if result:
                        walk_forward_results.append(result)
                except Exception as e:
                    print(f"Error in walk-forward test for {period_length} days: {e}")
            else:
                print(f"Skipping {period_length}-day test: not enough historical data")

        # Summarize walk-forward test results
        if walk_forward_results:
            print("\nWalk-Forward Test Summary:")
            print(
                f"{'Period':<10} {'Start Date':<12} {'End Date':<12} {'Actual Return':<15} {'Forecast':<15} {'Error %':<10} {'Percentile':<10} {'DD Calendar':<12} {'In 90% CI':<10}"
            )
            print("-" * 110)

            for result in walk_forward_results:
                period_desc = f"{result['period_length']}d"
                in_90_ci = "Yes" if result["in_90_interval"] else "No"
                in_50_ci = "Yes" if result["in_50_interval"] else "No"

                # Format percentages with the % symbol after the number
                actual_return_fmt = f"{result['actual_final_return']:.2f}%"
                median_forecast_fmt = f"{result['median_forecast']:.2f}%"
                percent_error_fmt = f"{result['percent_error']:.2f}%"
                percentile_fmt = f"{result['actual_percentile']:.1f}%"
                dd_calendar = f"{result['actual_dd_duration_calendar']} days"

                print(
                    f"{period_desc:<10} {result['test_start_date']:<12} {result['test_end_date']:<12} "
                    f"{actual_return_fmt:<15} {median_forecast_fmt:<15} {percent_error_fmt:<10} {percentile_fmt:<10} "
                    f"{dd_calendar:<12} {in_90_ci:<10}"
                )

            # Create a DataFrame for walk-forward test results
            wf_results_df = pd.DataFrame(walk_forward_results)

            # Round all numeric columns to 2 decimal places
            for column in wf_results_df.columns:
                if wf_results_df[column].dtype in ["float64", "float32"]:
                    wf_results_df[column] = wf_results_df[column].round(2)

            # Save walk-forward test results to CSV with symphony name prefixed
            wf_csv_path = os.path.join(
                output_dir, f"{clean_symphony_name}_walk_forward_results.csv"
            )
            wf_results_df.to_csv(wf_csv_path, index=False)
            print(f"\nWalk-forward test results saved to CSV: {wf_csv_path}")

            # Create a chart comparing actual vs forecast returns
            plt.figure(figsize=(10, 6))
            periods = [f"{r['period_length']}d" for r in walk_forward_results]
            actual_returns = [r["actual_final_return"] for r in walk_forward_results]
            forecast_returns = [r["median_forecast"] for r in walk_forward_results]

            x = range(len(periods))
            width = 0.35

            plt.bar(
                [i - width / 2 for i in x],
                actual_returns,
                width,
                label="Actual Return",
                color="green",
            )
            plt.bar(
                [i + width / 2 for i in x],
                forecast_returns,
                width,
                label="Forecast Return",
                color="blue",
            )

            plt.xlabel("Time Period")
            plt.ylabel("Cumulative Return (%)")
            plt.title(f"Actual vs Forecast Returns by Time Period - {symphony_name}")
            plt.xticks(x, periods)
            plt.grid(True, alpha=0.3)
            plt.legend()

            # Modified save path with symphony name prefixed
            comparison_chart_path = os.path.join(
                output_dir, f"{clean_symphony_name}_comparison.png"
            )
            plt.savefig(comparison_chart_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"Comparison chart saved to: {comparison_chart_path}")

    # Run rolling walk tests if selected
    if mode in ["2", "3"]:
        print("\n------------------------------------------")
        print("Rolling Walk-Forward Tests")
        print("------------------------------------------")

        # Get parameters for rolling walk test
        default_train_length = 504  # Default to 2 year training
        default_test_length = 252  # Default to 1 year testing
        default_step_size = 252  # Default to yearly steps

        train_length = input(
            f"Enter training period length in days (default: {default_train_length}): "
        )
        train_length = int(train_length) if train_length else default_train_length

        test_length = input(
            f"Enter test period length in days (default: {default_test_length}): "
        )
        test_length = int(test_length) if test_length else default_test_length

        step_size = input(f"Enter step size in days (default: {default_step_size}): ")
        step_size = int(step_size) if step_size else default_step_size

        # Check if we have enough data
        if len(daily_returns) < (train_length + test_length):
            print(
                f"Error: Not enough data for rolling walk test. Need at least {train_length + test_length} days."
            )
        else:
            try:
                # Run the rolling walk-forward test
                run_rolling_walk_forward_test(
                    date_strs,
                    daily_returns.tolist(),
                    train_length,
                    test_length,
                    output_dir,
                    clean_symphony_name,
                    step_size,
                )
            except Exception as e:
                print(f"Error in rolling walk-forward test: {e}")

    print("\nMonte Carlo analysis complete!")
    print(f"Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
