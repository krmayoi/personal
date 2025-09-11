import pandas as pd
import numpy as np

def detect_close_column(merged_df, base_ticker: str) -> str:
    # Prefer Adj Close if you want total-return-like behavior; use Close for strict price.
    candidates = [f"{base_ticker}_Adj Close", f"{base_ticker}_Close"]
    for c in candidates:
        if c in merged_df.columns:
            return c
    # Fallback: scan columns that start with ticker and include 'Close'
    for col in merged_df.columns:
        if col.startswith(f"{base_ticker}_") and "Close" in col:
            return col
    raise ValueError(f"No close price column found for ticker '{base_ticker}'")

def simulate_long_short_portfolio(
    merged_df: pd.DataFrame,
    dates,
    predictions,                     # 0 = Down (short), 1 = Up (long)
    base_ticker: str,
    starting_capital: float = 10000.0,
    trans_cost_bp: float = 10.0,     # 10 bps per trade leg (0.10%)
    annual_borrow_rate: float = 0.03,# 3% borrow on shorts (annualized)
    include_buy_hold: bool = True
):
    """
    Returns a dict with:
      - 'long_short': pd.Series of portfolio values for the long-short strategy
      - 'buy_hold': pd.Series of portfolio values (if include_buy_hold)
      - 'details': pd.DataFrame with returns, positions, costs
    """
    close_col = detect_close_column(merged_df, base_ticker)

    # Align and compute daily returns
    px = merged_df.loc[dates, close_col]
    daily_ret = px.pct_change().fillna(0.0).values

    # Build position series: +1 for pred=1, -1 for pred=0
    preds = np.asarray(predictions).astype(int)
    position = np.where(preds == 1, 1.0, -1.0)

    # Transaction costs: charged when position changes sign (rebalance day).
    # Cost modeled as 2 legs: out of prior position and into new one.
    # Effective round-trip cost = 2 * trans_cost_bp bps when sign flips.
    trans_cost_rate = trans_cost_bp / 10000.0
    pos_change = np.insert(np.diff(position), 0, 0.0)  # first day no prior position
    sign_flip = (pos_change != 0.0).astype(float)
    # Round-trip cost per flip:
    flip_cost = 2.0 * trans_cost_rate
    trans_costs = flip_cost * sign_flip  # as a fraction of capital

    # Borrow fee on short days (daily accrual)
    daily_borrow = annual_borrow_rate / 252.0
    borrow_costs = np.where(position < 0, daily_borrow, 0.0)

    # Strategy daily return before costs: exposure * asset return
    gross_ret = position * daily_ret

    # Net daily return after costs (costs reduce return)
    net_ret = gross_ret - trans_costs - borrow_costs

    # Compound
    values = [starting_capital]
    for r in net_ret:
        values.append(values[-1] * (1.0 + r))
    ls_series = pd.Series(values[1:], index=dates, name="long_short")

    results = {"long_short": ls_series}

    # Buy & Hold baseline
    if include_buy_hold:
        bh_values = [starting_capital]
        for r in daily_ret:
            bh_values.append(bh_values[-1] * (1.0 + r))
        results["buy_hold"] = pd.Series(bh_values[1:], index=dates, name="buy_hold")

    # Details for diagnostics
    details = pd.DataFrame(
        {
            "ret_asset": daily_ret,
            "position": position,
            "gross_ret": gross_ret,
            "trans_cost": trans_costs,
            "borrow_cost": borrow_costs,
            "net_ret": net_ret,
        },
        index=dates,
    )
    results["details"] = details
    return results
