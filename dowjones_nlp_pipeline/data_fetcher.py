import requests
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
