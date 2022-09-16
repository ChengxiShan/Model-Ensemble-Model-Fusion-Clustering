import pandas as pd

def matchTS(referrd_series,series):
    date_idx=list(referrd_series.index)
    matched_s=series.loc[(date_idx,slice(None)),:]
    return matched_s

def matchDF(referrd_series,df):
    date_idx=list(referrd_series.index)
    matched_df=df.loc[(date_idx,slice(None)),:]
    return matched_df