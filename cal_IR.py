import trans2df as tdf
import pandas as pd


def cal_ir(ts, true_ts):  # dual-index time series
    train_df = tdf.trans_df(ts)
    true_df = tdf.trans_df(true_ts)

    ic_list = []
    for i in range(len(train_df)):
        cur_ic = train_df.iloc[i, :].corr(true_df.iloc[i, :])
        ic_list.append(cur_ic)

    ic_list = pd.Series(ic_list)
    ir = ic_list.mean() / ic_list.std()

    return abs(ir)
