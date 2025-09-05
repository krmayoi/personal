from data_fetcher import DataFetcher
from config import DOW_JONES_URL, START_DATE, END_DATE

def main():
    # Create the fetcher
    fetcher = DataFetcher()

    # Step 1: Get Dow Jones tickers
    fetcher.get_dow_jones_tickers(url=DOW_JONES_URL)
    print(f"✅ Found {len(fetcher.tickers)} tickers: {fetcher.tickers}")

    # Step 2: Get historical market data
    fetcher.get_market_data(start=START_DATE, end=END_DATE)
    print(f"✅ Got returns for {len(fetcher.data)} tickers")

    # Optional: Preview one ticker's returns
    if fetcher.data:
        first_ticker = list(fetcher.data.keys())[0]
        print(f"\n📈 Sample data for {first_ticker}:")
        print(fetcher.data[first_ticker].head())

if __name__ == "__main__":
    main()

