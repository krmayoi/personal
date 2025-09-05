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
