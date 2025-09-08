import pandas as pd
import yfinance as yf
import numpy as np
from config import CIK_dict, DAYS_AFTER_FILING

def get_prices_after_filing(tickers, tenk_df):
    date_dict = {"Ticker": [], "Date": []}

    # Build filing date table
    for ticker in tickers:
        cik = CIK_dict[ticker]
        company_data = tenk_df[(tenk_df['CIK'] == str(cik)) & (tenk_df['Form Type'] == '10-K')]
        if company_data.empty:
            print(f"⚠️ No 10-K found for {ticker}")
            continue
        date = company_data.iloc[0]['Date Filed']
        date_dict['Ticker'].append(ticker)
        date_dict['Date'].append(date)

    date_df = pd.DataFrame(date_dict)
    date_df['Date'] = pd.to_datetime(date_df['Date'])

    # Download prices
    stock_prices_dict = {}
    for ticker in date_df['Ticker']:
        np.random.seed(1)
        start_date = date_df.loc[date_df['Ticker'] == ticker, 'Date'].iloc[0] + pd.DateOffset(days=1)
        end_date = start_date + pd.DateOffset(days=DAYS_AFTER_FILING)
        try:
            prices = yf.download(ticker, start=start_date, end=end_date)['Close']
            stock_prices_dict[ticker] = prices
        except Exception as e:
            print(f"⚠️ Failed to fetch prices for {ticker}: {e}")

    return pd.concat(stock_prices_dict) if stock_prices_dict else pd.DataFrame()
