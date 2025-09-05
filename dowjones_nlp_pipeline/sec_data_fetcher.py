# sec_data_fetcher.py
import requests
import pandas as pd
import time
from config import SEC_HEADERS  # we'll add this to config.py

class SECDataFetcher:
    def __init__(self, year):
        self.year = year
        self.data = None

    def get_master_index(self, form_type="10-K"):
        year_data = []
        for Q in ['QTR1', 'QTR2', 'QTR3', 'QTR4']:
            url = f"https://www.sec.gov/Archives/edgar/full-index/{self.year}/{Q}/master.idx"
            resp = requests.get(url, headers=SEC_HEADERS)

            # Only parse if we got plain text
            if resp.status_code == 200 and 'text/plain' in resp.headers.get('Content-Type', ''):
                lines = resp.text.splitlines()
                # Find header row
                header_idx = None
                for i, line in enumerate(lines):
                    if line.startswith("CIK|Company Name|Form Type|Date Filed|Filename"):
                        header_idx = i
                        break
                if header_idx is None:
                    print(f"⚠️ Could not find header in {Q}")
                    continue

                data_lines = lines[header_idx+1:]
                mydf = pd.DataFrame(
                    [row.split('|') for row in data_lines if row.strip()],
                    columns=lines[header_idx].split('|')
                )
                year_data.append(mydf)
            else:
                print(f"⚠️ Skipping {Q} — got {resp.status_code} {resp.headers.get('Content-Type')}")
            time.sleep(0.5)  # be nice to SEC

        if year_data:
            df = pd.concat(year_data, ignore_index=True)
            if form_type:
                df = df.loc[df['Form Type'] == form_type]
            self.data = df
        else:
            self.data = pd.DataFrame()
        return self
