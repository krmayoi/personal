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

# -----------------------------
# SEC Header
# -----------------------------
SEC_HEADERS = {'User-Agent': 'Sean Kim (krmayoi88@gmail.com)'}

# -----------------------------
# SEC Filings Path
# -----------------------------
SEC_FILINGS_PATH = "data/raw/sec_filings"

# -----------------------------
# CIK Dictionary
# -----------------------------
CIK_dict = {
    "AAPL": "320193",
    "MSFT": "789019",
    "UNH": "731766",
    "V": "1403161",
    "JNJ": "200406",
    "WMT": "104169",
    "JPM": "19617",
    "PG": "80424",
    "HD": "354950",
    "CVX": "93410",
    "KO": "21344",
    "DIS": "1744489",
    "CSCO": "858877",
    "VZ": "732712",
    "NKE": "320187",
    "MRK": "310158",
    "INTC": "50863",
    "CRM": "1108524",
    "MCD": "63908",
    "AXP": "4962",
    "AMGN": "318154",
    "HON": "773840",
    "CAT": "18230",
    "IBM": "51143",
    "GS": "886982",
    "BA": "12927",
    "MMM": "66740",
    "TRV": "86312",
    "WBA": "1618921"
}

# -----------------------------
# Days After Filing 10-K
# -----------------------------
DAYS_AFTER_FILING = 60

# -----------------------------
# ML Stock and Date Choice
# -----------------------------
BASE_TICKER = "JPM"
START_DATE = date(2010, 1, 1)
END_DATE = date(2024, 12, 31)
AUTO_ADJUST = False
