import pandas as pd


def model_merge(model_res, period, model_num):  # merge all model's res_series of a peroid as an df
    period_all = pd.DataFrame()
    model_nm = []
    for i in range(model_num):
        period_all = pd.concat([period_all, model_res[i][period - 1]], axis=1)
        model_nm.append('model' + str(i + 1))
    period_all.columns = model_nm
    period_all.index = model_res[0][period - 1].index
    return period_all
