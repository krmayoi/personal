# feature_engineering.py

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from data_fetcher import DataFetcher  # dynamically fetches DJIA tickers

def engineer_features_from_df(df, ticker):
    """Run technical feature engineering on a single ticker DataFrame."""
    if 'Adj Close' in df.columns:
        df['Ch_AdjClose'] = df['Adj Close'].diff()
    else:
        df['Ch_AdjClose'] = df['Close'].diff()

    df['Range'] = df['High'] - df['Low']
    df['RangeClose'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
    df['Open Higher'] = (df['Open'] > df['Close'].shift(1)).astype(int)

    for window in [7, 14]:
        df[f'Volume_{window}d_MA'] = df['Volume'].rolling(window).mean()
        df[f'Close_{window}d_MA'] = df['Close'].rolling(window).mean()
        df[f'Range_{window}d_MA'] = df['Range'].rolling(window).mean()
        df[f'RangeClose_{window}d_MA'] = df['RangeClose'].rolling(window).mean()
        df[f'Open Higher_{window}d_MA'] = df['Open Higher'].rolling(window).mean()

    df['VolumeUp'] = (df['Volume_7d_MA'] > df['Volume_14d_MA']).astype(int)
    df['CloseUp'] = (df['Close_7d_MA'] > df['Close_14d_MA']).astype(int)
    df['RangeUp'] = (df['Range_7d_MA'] > df['Range_14d_MA']).astype(int)

    df['CurrVol Up'] = (df['Volume'] > df['Volume_7d_MA']).astype(int)
    df['CurrClose Up'] = (df['Close'] > df['Close_7d_MA']).astype(int)
    df['CurrRange Up'] = (df['Range'] > df['Range_7d_MA']).astype(int)

    df['L14'] = df['Low'].rolling(14).min()
    df['H14'] = df['High'].rolling(14).max()
    df['SO'] = 100 * (df['Close'] - df['L14']) / (df['H14'] - df['L14'])
    df['R14'] = df['H14'] - df['L14']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df.add_prefix(f"{ticker}_")

def get_features_with_partner(base_ticker, start_date, end_date, auto_adjust=False):
    """Finds the most correlated Dow Jones ticker to base_ticker and engineers features for both."""
    np.random.seed(1)

    # Get DJIA tickers
    fetcher = DataFetcher().get_dow_jones_tickers()
    dj_tickers = fetcher.tickers
    if "DOW" in dj_tickers:
        dj_tickers.remove("DOW")
    if base_ticker not in dj_tickers:
        dj_tickers.append(base_ticker)

    # Bulk download with MultiIndex intact
    all_data = yf.download(
        dj_tickers,
        start=start_date,
        end=end_date,
        auto_adjust=auto_adjust,
        group_by='ticker'
    )

    # Build returns DataFrame for correlation
    returns = {}
    for t in dj_tickers:
        try:
            field = "Close" if auto_adjust else "Adj Close"
            series = all_data[t][field].pct_change()
            if isinstance(series, pd.Series):
                returns[t] = series
        except KeyError:
            pass

    returns = {k: v for k, v in returns.items() if not v.dropna().empty}
    returns_df = pd.DataFrame(returns).dropna()

    if base_ticker not in returns_df.columns:
        raise ValueError(f"Base ticker {base_ticker} has no return data — cannot compute correlation.")
    if len(returns_df.columns) < 2:
        raise ValueError("No other tickers with return data — cannot compute correlation.")

    correlations = returns_df.corr()[base_ticker].drop(base_ticker)
    most_corr_ticker = correlations.idxmax()

    # Slice clean single-ticker DataFrames from MultiIndex
    base_df = all_data[base_ticker].copy()
    partner_df = all_data[most_corr_ticker].copy()

    # Feature engineering
    base_features = engineer_features_from_df(base_df, base_ticker)
    partner_features = engineer_features_from_df(partner_df, most_corr_ticker)

    # Merge features
    merged_df = pd.concat([base_features, partner_features], axis=1, join='inner')

    # Flatten final merged columns if desired
    merged_df.columns = [col for col in merged_df.columns]

    return merged_df, most_corr_ticker
