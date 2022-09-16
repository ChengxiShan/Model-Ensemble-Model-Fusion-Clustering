import pandas as pd
import numpy as np
import LinearFusion as lf
import MatchTS as mts


def get_group_weight(train_labels, pred_labels, train_df, pred_df, true_rtn, weight_method='ic',
                     bt_rtn=None,open=None,close=None):
    true_rtn.name = 'true_res'
    n = len(train_labels.unique())
    train_wt_df = pd.DataFrame()
    pred_wt_df = pd.DataFrame()

    for i in range(n):
        cur_group_train_date = train_labels[train_labels.values == i]
        cur_group_pred_date = pred_labels[pred_labels.values == i]

        cur_train = mts.matchDF(cur_group_train_date, train_df)
        cur_pred = mts.matchDF(cur_group_pred_date, pred_df)

        train_len = len(cur_group_train_date)
        pred_len = len(cur_group_pred_date)

        cur_train_wt_df = pd.DataFrame()
        cur_pred_wt_df = pd.DataFrame()

        fusion = lf.LinearFusion(true_rtn,cur_train,models_train_bt_rtn=bt_rtn,open=open,close=close)

        if weight_method == 'ic':
            wt_list = fusion.ic_norm_weight()
        elif weight_method == 'ir':
            wt_list = fusion.ir_norm_weight()
        elif weight_method == 'lasso':
            wt_list = fusion.lasso_weight()
        elif weight_method=='bt_rtn':
            wt_list=fusion.bt_rtn_weight()
        else:
            wt_list = fusion.naive_avg_weight()

        wt_list = np.array(wt_list).reshape(1, len(wt_list))
        wt_list = pd.DataFrame(wt_list)
        wt_list.columns = train_df.columns

        print(wt_list)

        for i in range(train_len):
            cur_train_wt_df = pd.concat([cur_train_wt_df, wt_list], axis=0)
        cur_train_wt_df.index = cur_group_train_date.index
        train_wt_df = pd.concat([train_wt_df, cur_train_wt_df], axis=0)

        for j in range(pred_len):
            cur_pred_wt_df = pd.concat([cur_pred_wt_df, wt_list], axis=0)
        cur_pred_wt_df.index = cur_group_pred_date.index
        pred_wt_df = pd.concat([pred_wt_df, cur_pred_wt_df], axis=0)

    train_wt_df = train_wt_df.sort_index()
    pred_wt_df = pred_wt_df.sort_index()

    return train_wt_df, pred_wt_df
