from data_fetcher import DataFetcher
from portfolio_analysis import PortfolioAnalyzer
from config import DOW_JONES_URL, START_DATE, END_DATE

def main():
    # Step 1: Get Dow Jones tickers
    fetcher = DataFetcher().get_dow_jones_tickers(url=DOW_JONES_URL)
    print(f"✅ Found {len(fetcher.tickers)} tickers: {fetcher.tickers}")

    # Step 2: Get historical market data
    fetcher.get_market_data(start=START_DATE, end=END_DATE)
    print(f"✅ Got returns for {len(fetcher.data)} tickers")

    # Optional: Preview one ticker's returns
    if fetcher.data:
        first_ticker = list(fetcher.data.keys())[0]
        print(f"\n📈 Sample data for {first_ticker}:")
        print(fetcher.data[first_ticker].head())

    # Step 3: Run portfolio analysis
    analyzer = PortfolioAnalyzer(
        tickers=fetcher.tickers,
        start_date=START_DATE,
        end_date=END_DATE
    )
    analyzer.fetch_prices().run_overlapping_simulations()

    # Step 4: Show portfolio analysis results
    print("\n📊 Portfolio Analysis Results:")
    for period, data in analyzer.results.items():
        print(f"\nPeriod: {period}")
        print(data['DF'].head())

if __name__ == "__main__":
    main()
