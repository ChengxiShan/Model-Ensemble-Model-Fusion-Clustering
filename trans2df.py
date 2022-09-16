import pandas as pd


def trans_df(series):  # transfer dual-index time series as a df
    series = series.reset_index()
    series.columns = ['datetime', 'asset', 'val']  # index：datetime cols：asset—symbol
    df = pd.pivot_table(series, index=series.datetime, columns=series.asset, dropna=False)
    return df
