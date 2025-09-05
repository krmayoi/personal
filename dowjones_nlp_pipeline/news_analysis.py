import requests
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

class NewsAnalyzer:
    """
    Fetches headlines from Finviz and scores sentiment using VADER.
    """

    def __init__(self, tickers):
        """
        Parameters
        ----------
        tickers : list
            List of ticker symbols to fetch headlines for.
        """
        self.tickers = tickers
        self.header = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/71.0.3578.98 Safari/537.36'
            )
        }
        self.sentiment = SentimentIntensityAnalyzer()

    def fetch_headlines(self, ticker):
        """
        Fetch headlines and sentiment for a single ticker.

        Returns
        -------
        pd.DataFrame
            Columns: Ticker, Title, Date, Time, Compound
        """
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        try:
            html_data = requests.get(url, headers=self.header, timeout=10)
            html_data.raise_for_status()
        except requests.RequestException as e:
            print(f"⚠️ Failed to fetch headlines for {ticker}: {e}")
            return pd.DataFrame(columns=["Ticker", "Title", "Date", "Time", "Compound"])

        soup = BeautifulSoup(html_data.text, 'lxml')
        headlines = soup.find_all('tr', class_='cursor-pointer has-label')

        data = {
            "Ticker": [],
            "Title": [],
            "Date": [],
            "Time": [],
            "Compound": [],
        }

        lastdate = None
        for headline in headlines:
            title_tag = headline.find('a', class_='tab-link-news')
            if not title_tag:
                continue
            title = title_tag.text.strip()

            datetime_html = headline.find('td', width="130", align="right")
            date, time = None, None
            if datetime_html:
                datetime_str = datetime_html.text.strip()
                if 'Today' in datetime_str:
                    date = 'Today'
                    time = datetime_str.split(' ', 1)[1]
                    lastdate = date
                elif '-' in datetime_str:
                    date, time = datetime_str.split(' ', 1)
                    lastdate = date
                else:
                    date = lastdate
                    time = datetime_str

            compound = self.sentiment.polarity_scores(title)['compound']

            data["Ticker"].append(ticker)
            data["Title"].append(title)
            data["Date"].append(date)
            data["Time"].append(time)
            data["Compound"].append(compound)

        return pd.DataFrame(data)

    def fetch_all(self):
        """
        Fetch headlines for all tickers in self.tickers.
        """
        all_data = []
        for ticker in self.tickers:
            df = self.fetch_headlines(ticker)
            if not df.empty:
                all_data.append(df)
        return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

    def average_compound_scores(self):
        """
        Calculate the average compound sentiment score for each ticker
        using all fetched headlines.

        Returns
        -------
        pd.Series
            Index: Ticker, Value: Average compound score
        """
        all_headlines = self.fetch_all()
        if all_headlines.empty:
            return pd.Series(dtype=float)

        return all_headlines.groupby("Ticker")["Compound"].mean()
