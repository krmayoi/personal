import pandas as pd
import numpy as np
import yfinance as yf
from config import CIK_dict, DAYS_AFTER_FILING

def get_prices_after_filing(tickers, tenk_df):
    """Fetch daily Close prices for DAYS_AFTER_FILING days after each 10-K filing."""
    date_dict = {"Ticker": [], "Date": []}

    # Build filing date table
    for ticker in tickers:
        cik = CIK_dict[ticker]
        company_data = tenk_df[(tenk_df['CIK'] == str(cik)) & (tenk_df['Form Type'] == '10-K')]
        if company_data.empty:
            print(f"⚠️ No 10-K found for {ticker}")
            continue
        date = company_data.iloc[0]['Date Filed']
        date_dict["Ticker"].append(ticker)
        date_dict["Date"].append(date)

    date_df = pd.DataFrame(date_dict)
    date_df["Date"] = pd.to_datetime(date_df["Date"])

    # Download Close prices
    stock_prices_dict = {}
    for ticker in date_df["Ticker"]:
        np.random.seed(1)
        start_date = date_df.loc[date_df["Ticker"] == ticker, "Date"].iloc[0] + pd.DateOffset(days=1)
        end_date = start_date + pd.DateOffset(days=DAYS_AFTER_FILING)
        try:
            prices = yf.download(ticker, start=start_date, end=end_date)["Close"]
            stock_prices_dict[ticker] = prices
        except Exception as e:
            print(f"⚠️ Failed to fetch prices for {ticker}: {e}")

    return pd.concat(stock_prices_dict) if stock_prices_dict else pd.DataFrame()

def calculate_variance(stock_prices_dict):
    """Calculate variance of daily returns for each ticker using Close prices."""
    variance_dict = {}
    for ticker, prices in stock_prices_dict.items():
        daily_chng = prices.pct_change().dropna()
        mean_return = daily_chng.mean()
        sq_deviations = (daily_chng - mean_return) ** 2
        variance = sq_deviations.mean()
        variance_dict[ticker] = variance
    return pd.DataFrame(list(variance_dict.items()), columns=["Ticker", "Variance"])
