import requests
import os
import json
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer
import yfinance as yf
import pandas as pd
import time
from datetime import date
from config import DOW_JONES_URL, START_DATE, END_DATE

class DataFetcher:
    def __init__(self):
        self.tickers = []
        self.sentiment = SentimentIntensityAnalyzer()
        self.data = {}

    def get_dow_jones_tickers(self, url=DOW_JONES_URL):
        headers = {'User-Agent': 'Mozilla/5.0'}
        html_data = requests.get(url, headers=headers)
        soup = BeautifulSoup(html_data.text, 'lxml')

        classes = [f'row-{x}' for x in range(2, 32)]
        tickers = []
        for class_name in classes:
            row = soup.find('tr', class_=class_name)
            if row:
                ticker = row.find('td', class_='column-1').text.strip()
                tickers.append(ticker)

        self.tickers = sorted(tickers)
        return self
    
    def save_tickers(self, save_dir="data/raw", filename="djia_tickers.json"):
        """Save the current list of DJIA tickers to a file in save_dir."""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, filename)

        try:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self.tickers, f, indent=2)
            print(f"✅ Saved {len(self.tickers)} tickers to {save_path}")
        except Exception as e:
            print(f"⚠️ Error saving tickers: {e}")

    def get_market_data(self, start=START_DATE, end=END_DATE):
        import numpy as np
        np.random.seed(1)

        # Remove problematic ticker before fetching
        if "DOW" in self.tickers:
            self.tickers.remove("DOW")

        stock_returns = {}
        for ticker in self.tickers:
            try:
                dailyprc = yf.download(
                    ticker, start, end,
                    interval="1d", progress=False, threads=False,
                    auto_adjust=True  # adjusted Close
                )
                if dailyprc.empty:
                    print(f"⚠️ No data for {ticker}, skipping.")
                    continue
                dailyrets = dailyprc.pct_change().dropna()
                stock_returns[ticker] = dailyrets
                time.sleep(1)  # avoid hammering Yahoo
            except Exception as e:
                print(f"❌ Failed to fetch {ticker}: {e}")

        self.data = stock_returns
        return self
    
    def save_market_data(self, save_path="data/raw/djia_prices.pkl"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        try:
            pd.to_pickle(self.data, save_path)
            print(f"✅ Saved market data for {len(self.data)} tickers to {save_path}")
        except Exception as e:
            print(f"⚠️ Error saving market data: {e}")

    def get_yearly_returns(self, year, drop_tickers_with_no_data=('DOW',)):
        """
        Fetch daily returns and absolute adjusted price changes for all tickers for a given year.
        Uses auto_adjust=True so 'Close' is adjusted (no 'Adj Close' column needed).

        Returns
        -------
        dict: {ticker: DataFrame with ['Close', 'Ret_Close', 'Ch_Close', ...]}
        """
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)

        stock_returns = {}
        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    auto_adjust=True,
                    progress=False
                )

                if df.empty or 'Close' not in df.columns:
                    print(f"⚠️ No usable data for {ticker} in {year}")
                    continue

                out = pd.DataFrame(index=df.index)
                out['Close'] = df['Close']
                out['Ret_Close'] = out['Close'].pct_change()
                out['Ch_Close'] = out['Close'].diff()

                if 'Volume' in df.columns:
                    out['Volume'] = df['Volume']

                stock_returns[ticker] = out.dropna()

            except Exception as e:
                print(f"⚠️ Failed to fetch {ticker} for {year}: {e}")

        # Drop problematic tickers
        for bad in drop_tickers_with_no_data:
            if bad in stock_returns:
                del stock_returns[bad]

        return stock_returns
