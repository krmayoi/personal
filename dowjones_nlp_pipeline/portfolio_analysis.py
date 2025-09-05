import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
from scipy.optimize import minimize

class PortfolioAnalyzer:
    def __init__(self, tickers, start_date, end_date, num_simul=50000, max_weight=0.10):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.num_simul = num_simul
        self.max_weight = max_weight
        self.daily_prices = None
        self.results = {}
        self.year_ranges = []  # store ranges for reuse

    def fetch_prices(self):
        self.daily_prices = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )['Close'].dropna()
        return self

    def run_overlapping_simulations(self):
        start_years = range(2010, 2022)
        end_years = range(2012, 2024)
        self.year_ranges = [f"{s}:{e}" for s, e in zip(start_years, end_years)]
        for s, e in zip(start_years, end_years):
            self._simulate_for_range(s, e)
        return self.results

    def _simulate_for_range(self, start_year, end_year):
        start_date_range = f"{start_year}-01-01"
        end_date_range = f"{end_year}-12-31"

        prices = self.daily_prices.loc[start_date_range:end_date_range]
        monthly_prices = prices.resample('M').last()
        monthly_returns = monthly_prices.pct_change()

        mu_vec = np.array(monthly_returns.mean() * 12)
        cov_mat = np.array(monthly_returns.cov() * 12)
        N = len(self.tickers)

        port_expret = np.empty(self.num_simul)
        port_var = np.empty(self.num_simul)
        weights = np.empty((self.num_simul, N))
        np.random.seed(100)

        for i in range(self.num_simul):
            temp = np.random.rand(N)
            weights[i] = temp / temp.sum()
            port_expret[i] = np.dot(weights[i].T, mu_vec)
            port_var[i] = np.dot(weights[i].T, np.dot(cov_mat, weights[i]))

        port_sd = np.sqrt(port_var)

        def portsd_func(w, covmat):
            return np.sqrt(np.dot(w.T, np.dot(covmat, w)))

        sd_weights = weights[port_sd.argmin()]
        optweights = pd.DataFrame(sd_weights, index=self.tickers, columns=['Min_Var_Sim'])

        tenyr_yield = pdr.DataReader('DGS10', 'fred', start_date_range, end_date_range).dropna()
        rf = tenyr_yield.iloc[0][0] / 100
        sharpes = (port_expret - rf) / port_sd
        maxSR_weights = weights[sharpes.argmax()]
        optweights['Max_SR_Sim'] = maxSR_weights

        Sinv = np.linalg.inv(cov_mat)
        topmat = np.dot(Sinv, mu_vec - rf)
        denom = np.dot(np.ones(N), np.dot(Sinv, mu_vec - rf))
        maxSR_act_weights = topmat / denom
        optweights['Max_SR_Act'] = maxSR_act_weights

        def neg_sharpe(w, muret, covmat, rf):
            return -1 * ((np.dot(w.T, mu_vec) - rf) / portsd_func(w, covmat))

        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, self.max_weight) for _ in range(N)]
        initial_weights = np.ones(N) / N

        optimized_results = minimize(
            neg_sharpe, initial_weights,
            args=(mu_vec, cov_mat, rf),
            method='SLSQP',
            constraints=constraints,
            bounds=bounds
        )
        optweights['Max_SR_Constr'] = optimized_results.x.round(6)

        TwoS = 2 * np.array(cov_mat)
        Leftside = np.insert(TwoS, N, np.ones(N), axis=0)
        Lastcol = np.ones(N)
        Lastcol = np.append(Lastcol, 0)
        Amat = np.insert(Leftside, N, Lastcol, axis=1)
        b_vec = np.zeros(N)
        b_vec = np.append(b_vec, 1)
        Ainv = np.linalg.inv(Amat)
        gmv_weights = np.dot(Ainv, b_vec)[0:N]
        optweights['GMV_Act'] = gmv_weights

        self.results[f"{start_year}:{end_year}"] = {'DF': optweights}

    def backtest_max_sharpe(self, lookahead=3):
        """Backtest analytical Max Sharpe weights out-of-sample."""
        results = []
        N = len(self.tickers)

        for s in range(len(self.year_ranges) - lookahead):
            training_period = self.year_ranges[s]
            investment_period = self.year_ranges[s + lookahead]

            # Previous weights from training period
            current_weights = self.results[training_period]['DF']['Max_SR_Act'].values

            # Get next period's prices
            start_year_next, end_year_next = map(int, investment_period.split(":"))
            start_date_range_next = f"{start_year_next}-01-01"
            end_date_range_next = f"{end_year_next}-12-31"

            DJ_DailyAdjClose_next = yf.download(
                self.tickers,
                start=start_date_range_next,
                end=end_date_range_next,
                auto_adjust=True
            )['Close'].dropna()

            DJ_MonthlyAdjClose_next = DJ_DailyAdjClose_next.resample('M').last()
            DJ_MonthlyReturns_next = DJ_MonthlyAdjClose_next.pct_change()

            start_price = DJ_MonthlyAdjClose_next.iloc[0]
            end_price = DJ_MonthlyAdjClose_next.iloc[-1]
            relative_rets = (end_price - start_price) / start_price

            cumulative_returns_current = np.dot(relative_rets, current_weights)

            # New Max Sharpe for next period
            tenyr_yield = pdr.DataReader('DGS10', 'fred', start_date_range_next, end_date_range_next).dropna()
            rf_next = tenyr_yield.iloc[0][0] / 100
            mu_vec_next = np.array(DJ_MonthlyReturns_next.mean() * 12)
            cov_mat_next = np.array(DJ_MonthlyReturns_next.cov() * 12)
            Sinv_next = np.linalg.inv(cov_mat_next)
            topmat_next = np.dot(Sinv_next, mu_vec_next - rf_next)
            denom_next = np.dot(np.ones(N), np.dot(Sinv_next, mu_vec_next - rf_next))
            weights_next = topmat_next / denom_next

            cumulative_returns_next = np.dot(relative_rets, weights_next)

            results.append({
                'Training Period': training_period,
                'Investment Period': investment_period,
                'Prev Max Sharpe Return': cumulative_returns_current,
                'New Max Sharpe Return': cumulative_returns_next
            })

        return pd.DataFrame(results)
