# 📊 Dow Jones NLP & Portfolio Analysis Pipeline

## Overview
This project is an **end‑to‑end data pipeline** that integrates **financial market data**, **SEC 10‑K filings**, and **news sentiment analysis** to extract actionable insights from the Dow Jones Industrial Average (DJIA) constituents.

It combines:
- **Data engineering** — automated retrieval, cleaning, and storage of multi‑source datasets
- **Natural Language Processing (NLP)** — textual analysis of SEC filings using domain‑specific dictionaries
- **Portfolio analytics** — in‑sample simulations, out‑of‑sample backtesting, and risk metrics
- **Reproducible architecture** — modular, config‑driven design for easy extension

The next phase will extend this into **machine learning models** to predict market behaviour from textual and quantitative features.

---

## ✨ Features

### 1. Data Acquisition
- **Dow Jones tickers** fetched dynamically from a configurable source
- **Historical market data** via Yahoo Finance
- **SEC 10‑K filings** downloaded and stored locally
- **News headlines** scraped for sentiment analysis

### 2. Portfolio Analysis
- In‑sample simulations with overlapping periods
- Out‑of‑sample backtesting of the **Max Sharpe** portfolio
- **Value at Risk (VaR)** analysis for equal‑weighted portfolios

### 3. NLP on SEC Filings
- **Uncertainty** score (LM dictionary)
- **Tone** score (positive vs. negative LM words)
- **FOG Index** (text complexity)
- **Flesch Reading Ease** (readability)
- Efficient, cached syllable counting for large documents

### 4. Sentiment Analysis
- Headline sentiment scoring using VADER
- Aggregated compound sentiment per ticker

### 5. Machine Learning Models
- Feature engineering from both quantitative market data and partner-ticker correlations (`feature_engineering.py`)
- Config-driven base ticker selection -- swap `BASE_TICKER` in `config.py` to train on any ticker
- Automated model tuning with `RandomizedSearchCV` + `GridSearchCV` for:
    - Decision Trees
    - Random Forests
    - Gradient Boosting
- Model ranking by combined Accuracy, Precision, and scaled MSE
- Portfolio simulation of ML predictions:
    - Long-short strategy
    - Buy-and-hold benchmark
    - Summary stats: total return, annualized return/volatility, max drawdown
  
--- 

## 🧠 ML Workflow

1. Feature Engineering
- Pulls base ticker + most correlated partner ticker
- Generates rolling technical indicators, volume/price trends, and stochastic oscillators
- Merges features into a single aligned dataset
2. Train / Holdout Split
- Date‑based or fraction‑based split via `split_model_holdout()`
- Holdout period configurable in `ml_train.py` for forward‑testing
3. Model Selection & Tuning
- Hyperparameter search for each model
- Best models stored in `selector.tuned_models`
4. Evaluation & Simulation
- Metrics on holdout set
- Portfolio backtest with realistic frictions
- Equity curve plots and performance summaries

---

## 🛠️ Tech Stack

**Languages & Libraries**
- Python 3.11.9
- `pandas`, `numpy`, `yfinance`, `nltk`, `scikit-learn` (ML phase)
- Custom modules: `data_fetcher`, `portfolio_analysis`, `news_analysis`, `sec_data_fetcher`, `text_metrics`

**Data Sources**
- Yahoo Finance (market data)
- SEC EDGAR (10‑K filings)
- News APIs / scraping

**Architecture**
- Modular, class‑based design
- Config‑driven parameters (`config.py`)
- Clear separation of concerns for maintainability

**Setup**
- **Note:** This project was developed and tested on Python 3.11.9.  
- Using a different Python version may cause dependency or compatibility issues.

---

## 📂 Project Structure
```
. 
  ├── main.py                   # Orchestrates the full pipeline 
  ├── config.py                 # Centralized configuration 
  ├── data_fetcher.py           # Market data retrieval 
  ├── portfolio_analysis.py     #  Portfolio simulation & risk metrics 
  ├── news_analysis.py          # Headline sentiment analysis 
  ├── sec_data_fetcher.py       # SEC filings retrieval & filtering 
  ├── text_metrics.py           # NLP metrics on 10-K filings 
  ├── data/ 
  │ ├── raw/                    # Unprocessed data (filings, raw CSVs) 
  │ ├── processed/              # Cleaned datasets & metrics 
  │ └── reference/              # LM dictionaries
  ├── ml_train.py               # ML training, evaluation, and portfolio simulation
  ├── feature_engineering.py    # Builds features from base + partner ticker
  ├── data_splits.py            # Train/holdout split logic
  ├── model_selection.py        # Model tuning, ranking, and export
  ├── portfolio_sim.py          # Long-short and buy-hold simulation
  └── README.md                 # Project documentation
```
---

## 🚀 How to Run

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/dowjones_nlp_pipeline.git
   cd dowjones_nlp_pipeline
   
2. Install dependencies
   pip install -r requirements.txt
   
3. Configure parameters in config.py
  - Date ranges (START_DATE, END_DATE, PREDICTION_YEAR)
  - SEC filings path
  - Dow Jones tickers URL
  - Base Ticker choice

4. Run the pipeline
   python main.py

## 📜 License
This project is for educational and portfolio purposes. Data sources are subject to their respective terms of use.

## 👤 About Me
KRMayoi - Data science & analytics enthusiast, focused on reproducible, modular pipelines and the intersection of NLP with financial markets.
