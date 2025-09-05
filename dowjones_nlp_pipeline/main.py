from data_fetcher import DataFetcher
from portfolio_analysis import PortfolioAnalyzer
from news_analysis import NewsAnalyzer
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

    # Step 3: Run portfolio analysis (in-sample)
    analyzer = PortfolioAnalyzer(
        tickers=fetcher.tickers,
        start_date=START_DATE,
        end_date=END_DATE
    )
    analyzer.fetch_prices().run_overlapping_simulations()

    print("\n📊 Portfolio Analysis Results (In-Sample):")
    for period, data in analyzer.results.items():
        print(f"\nPeriod: {period}")
        print(data['DF'].head())

    # Step 4: Backtest Max Sharpe portfolio (out-of-sample)
    backtest_df = analyzer.backtest_max_sharpe()
    print("\n📈 Backtest Results (Out-of-Sample):")
    print(backtest_df.to_string(index=False))

    # Step 5: Value at Risk (VaR) analysis
    var_df = analyzer.var_analysis()
    print("\n📉 Value at Risk Analysis (Equal-Weighted Portfolio):")
    print(var_df)

    # Step 6: Headline sentiment analysis
    news_analyzer = NewsAnalyzer(fetcher.tickers)
    headline_df = news_analyzer.fetch_all()
    print("\n📰 Headline Sentiment Data:")
    print(headline_df.head())

if __name__ == "__main__":
    main()
