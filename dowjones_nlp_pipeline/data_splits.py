def split_model_holdout(merged_df, base_ticker, partner, holdout_year=2023, train_start=2010, train_end=2022):
    """Split merged_df into model and holdout sets with aligned X and y."""
    
    # Holdout set
    holdout_df = merged_df[merged_df.index.year == holdout_year]
    Y_holdout = (holdout_df[f"{base_ticker}_Ch_AdjClose"].shift(-1) > 0).astype(int)
    X_holdout = holdout_df[[
        f"{base_ticker}_RangeClose",
        f"{base_ticker}_Open Higher",
        f"{base_ticker}_VolumeUp",
        f"{base_ticker}_CloseUp",
        f"{base_ticker}_RangeUp",
        f"{base_ticker}_CurrVol Up",
        f"{base_ticker}_CurrClose Up",
        f"{base_ticker}_CurrRange Up",
        f"{base_ticker}_SO",
        f"{base_ticker}_R14",
        f"{partner}_RangeClose",
        f"{partner}_Open Higher",
        f"{partner}_VolumeUp",
        f"{partner}_CloseUp",
        f"{partner}_RangeUp",
        f"{partner}_CurrVol Up",
        f"{partner}_CurrClose Up",
        f"{partner}_CurrRange Up",
        f"{partner}_SO",
        f"{partner}_R14"
    ]].copy()
    X_holdout = X_holdout.dropna()
    Y_holdout = Y_holdout.loc[X_holdout.index]

    # Model set
    model_df = merged_df[(merged_df.index.year >= train_start) & (merged_df.index.year <= train_end)]
    Y_model = (model_df[f"{base_ticker}_Ch_AdjClose"].shift(-1) > 0).astype(int)
    X_model = model_df[X_holdout.columns].copy()  # ensures same feature order
    X_model = X_model.dropna()
    Y_model = Y_model.loc[X_model.index]

    return X_model, Y_model, X_holdout, Y_holdout
