import os
import time
import requests
import pandas as pd
from config import SEC_HEADERS, CIK_dict, SEC_FILINGS_PATH

class SECDataFetcher:
    def __init__(self, year):
        self.year = year
        self.data = pd.DataFrame()

    def get_master_index(self, form_type="10-K"):
        """Fetch the SEC master index for the given year and form type."""
        year_data = []
        for Q in ['QTR1', 'QTR2', 'QTR3', 'QTR4']:
            url = f"https://www.sec.gov/Archives/edgar/full-index/{self.year}/{Q}/master.idx"
            resp = requests.get(url, headers=SEC_HEADERS)

            if resp.status_code == 200:
                # Always try to decode as text, even if Content-Type is application/octet-stream
                try:
                    text = resp.text
                except UnicodeDecodeError:
                    # Fallback if .text fails
                    text = resp.content.decode('latin-1', errors='ignore')

                lines = text.splitlines()

                # Find header row dynamically
                header_idx = None
                for i, line in enumerate(lines):
                    if line.startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
                        header_idx = i
                        break

                if header_idx is None:
                    print(f"⚠️ Could not find header in {Q}")
                    continue

                data_lines = lines[header_idx + 1:]
                mydf = pd.DataFrame(
                    [row.split('|') for row in data_lines if row.strip()],
                    columns=lines[header_idx].split('|')
                )
                year_data.append(mydf)
            else:
                print(f"⚠️ Skipping {Q} — got HTTP {resp.status_code}")

            time.sleep(0.5)  # be nice to SEC

        if year_data:
            df = pd.concat(year_data, ignore_index=True)
            if form_type:
                df = df.loc[df['Form Type'] == form_type]
            self.data = df
        else:
            self.data = pd.DataFrame()

        return self


    def filter_to_dow_jones(self):
        """Filter the data to only include companies in CIK_dict."""
        if self.data.empty:
            print("⚠️ No SEC data loaded yet.")
            return self
        cik_values = set(CIK_dict.values())
        self.data = self.data[self.data['CIK'].isin(cik_values)]
        return self

    def build_url_df(self):
        """Build a DataFrame of tickers and full SEC filing URLs."""
        url_dict = {"Ticker": [], "URL": []}
        for ticker, cik in CIK_dict.items():
            company_data = self.data[self.data['CIK'] == str(cik)]
            if company_data.empty:
                print(f"⚠️ No filing found for {ticker}")
                continue
            filename = company_data.iloc[0]['Filename']
            url = f"https://www.sec.gov/Archives/{filename}"
            url_dict['Ticker'].append(ticker)
            url_dict['URL'].append(url)
        return pd.DataFrame(url_dict)

    def download_filings(self, url_df):
        """Download filings from the given URL DataFrame into SEC_FILINGS_PATH."""
        os.makedirs(SEC_FILINGS_PATH, exist_ok=True)
        for _, row in url_df.iterrows():
            ticker = row['Ticker']
            filing_url = row['URL']
            try:
                resp = requests.get(filing_url, headers=SEC_HEADERS)
                if resp.status_code == 200:
                    save_path = os.path.join(SEC_FILINGS_PATH, f"{ticker}_10K.txt")
                    with open(save_path, "w", encoding="utf-8") as f:
                        f.write(resp.text)
                    print(f"✅ Saved {ticker} 10-K")
                else:
                    print(f"❌ Failed to download {ticker} — HTTP {resp.status_code}")
            except Exception as e:
                print(f"⚠️ Error downloading {ticker}: {e}")
            time.sleep(0.5)  # avoid hammering SEC

