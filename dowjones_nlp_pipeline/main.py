from data_fetcher import DataFetcher
from portfolio_analysis import PortfolioAnalyzer
from news_analysis import NewsAnalyzer
from sec_data_fetcher import SECDataFetcher
from text_metrics import analyze_filings
from price_after_filing import get_prices_after_filing, calculate_variance  # <-- now imports both
from config import DOW_JONES_URL, START_DATE, END_DATE, PREDICTION_YEAR, DAYS_AFTER_FILING
import nltk


def main():
    # Step 1: Get Dow Jones tickers
    fetcher = DataFetcher().get_dow_jones_tickers(url=DOW_JONES_URL)
    print(f"✅ Found {len(fetcher.tickers)} tickers: {fetcher.tickers}")

    # Step 2: Get historical market data (training set)
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

    # Step 7: Average compound sentiment scores
    avg_scores = news_analyzer.average_compound_scores().round(3)
    print("\n📊 Average Compound Sentiment Scores:")
    for ticker, score in avg_scores.items():
        print(f"{ticker} | Average Compound Score = {score}")

    # Step 8: Fetch prediction year data for forward testing
    prediction_data = fetcher.get_yearly_returns(PREDICTION_YEAR)
    print(f"\n📅 Prediction Year {PREDICTION_YEAR} Data:")
    for ticker, df in prediction_data.items():
        print(f"{ticker}: {len(df)} trading days, last date = {df.index.max().date()}")

    # Step 9: Fetch 10-K filings for 2020
    sec_fetcher = SECDataFetcher(year=2020).get_master_index(form_type="10-K")
    print(f"\n📄 Found {len(sec_fetcher.data)} 10-K filings in {sec_fetcher.year}")
    print(sec_fetcher.data.head())

    # Step 10: Filter to Dow Jones companies
    sec_fetcher.filter_to_dow_jones()

    # Step 11: Build URL DataFrame for filings
    url_df = sec_fetcher.build_url_df()
    print(f"\n🔗 Built {len(url_df)} filing URLs for Dow Jones tickers")
    print(url_df.head())

    # Step 12: Download filings into data/raw/sec_filings/
    sec_fetcher.download_filings(url_df)

    # Step 13: Run integrated text metrics (Uncertainty, Tone, FOG, Flesch)
    nltk.download('cmudict')
    pronouncing_dict = nltk.corpus.cmudict.dict()

    metrics_df = analyze_filings(fetcher.tickers, pronouncing_dict)
    print("\n📊 Textual Analysis Metrics:")
    print(metrics_df.sort_values(by="Uncertainty", ascending=False))

    # Step 14: Get daily stock prices after 10-K filing (Close prices)
    all_stock_prices = get_prices_after_filing(fetcher.tickers, sec_fetcher.data)
    print(f"\n📈 Daily stock prices {DAYS_AFTER_FILING} days after 10-K filing:")
    print(all_stock_prices.head(15))

    # Step 15: Calculate volatility (variance) over that period
    # Convert MultiIndex DataFrame to dict of Series
    stock_prices_dict = {ticker: all_stock_prices.loc[ticker] for ticker in all_stock_prices.index.levels[0]}
    variance_df = calculate_variance(stock_prices_dict)
    print(f"\n📊 Variance of daily returns {DAYS_AFTER_FILING} days after filing:")
    print(variance_df.sort_values(by="Variance", ascending=False))


if __name__ == "__main__":
    main()

