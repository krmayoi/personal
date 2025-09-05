from datetime import date

# -----------------------------
# Data Sources
# -----------------------------
# URL for scraping Dow Jones tickers
DOW_JONES_URL = "https://bullishbears.com/dow-jones-stocks-list/"

# -----------------------------
# Date Range Defaults
# -----------------------------
START_DATE = date(2010, 1, 1)
END_DATE = date(2023, 12, 31)

# -----------------------------
# File Paths
# -----------------------------
RAW_DATA_PATH = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"

# -----------------------------
# Prediction Year
# -----------------------------
PREDICTION_YEAR = 2024
