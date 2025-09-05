import requests
from bs4 import BeautifulSoup
from nltk.sentiment import SentimentIntensityAnalyzer

class DataFetcher:
    def __init__(self):
        self.tickers = []
        self.sentiment = SentimentIntensityAnalyzer()

    def get_dow_jones_tickers(self, url=DOW_JONES_URL):
        headers = {'User-Agent': 'Mozilla/5.0'}
        html_data = requests.get(url, headers=headers)
        soup = BeautifulSoup(html_data.text, 'lxml')

        classes = []
        for x in range(2, 32):
            row_class = f'row-{x}'
            classes.append(row_class)

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
                dailyprc = yf.download(ticker, start, end, interval="1d", progress=False, threads=False)
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
