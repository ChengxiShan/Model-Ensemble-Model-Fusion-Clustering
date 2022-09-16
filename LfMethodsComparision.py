import pandas as pd
from LinearFusion import LinearFusion
from BackTest import BackTest
import matplotlib.pyplot as plt
import datetime


def test_method(train_data,test_data,true_rtn, n_splits, method_nm,models_train_bt_rtn=None):
    train_pred_list = []
    test_pred_list = []
    weight_list = []
    for i in range(n_splits):
        lf = LinearFusion(true_rtn,train_data[i],test_data[i],models_train_bt_rtn[i])
        if method_nm == 'ic':
            test_pred_list.append(lf.ic_test_pred())
            train_pred_list.append(lf.ic_train_pred())
            weight_list.append(lf.ic_norm_weight())
        if method_nm == 'na':
            test_pred_list.append(lf.na_test_pred())
            train_pred_list.append(lf.na_train_pred())
            weight_list.append(lf.naive_avg_weight())
        if method_nm == 'ir':
            test_pred_list.append(lf.ir_test_pred())
            train_pred_list.append(lf.ir_train_pred())
            weight_list.append(lf.ir_norm_weight())
        if method_nm == 'lasso':
            test_pred_list.append(lf.lasso_test_pred())
            train_pred_list.append(lf.lasso_train_pred())
            weight_list.append(lf.lasso_weight())
        if method_nm == 'er':
            test_pred_list.append(lf.ew_test_pred())
            train_pred_list.append(lf.ew_train_pred())
            weight_list.append(lf.error_norm_weight())
        if method_nm=='bt_rtn':
            test_pred_list.append((lf.bt_test_pred()))
            train_pred_list.append((lf.bt_train_pred()))
            weight_list.append(lf.bt_rtn_weight())
    test_pred_rtn = pd.concat([pred for pred in test_pred_list],axis=0)
    train_pred_rtn = pd.concat([pred for pred in train_pred_list],axis=0)
    return_items = [train_pred_rtn, test_pred_rtn, weight_list]
    return_id = ['train_pred_rtn', 'test_pred_rtn', 'weight_list']
    weight_df=pd.DataFrame(weight_list)
    weight_df.columns=train_data[0].columns
    weight_df.index=['period'+str(x) for x in range(6)]
    timestamp = datetime.datetime.now()
    weight_df.to_csv(method_nm+'_weight_df_'+str(timestamp)+'.csv')
    return dict(zip(return_id, return_items))


def compare_methods(train_data,test_data,true_rtn, n_splits, method_list, open, close,models_train_bt_rtn=None):
    bt_list = []
    for method in method_list:
        pred = test_method(train_data,test_data, true_rtn, n_splits, method,models_train_bt_rtn)
        bt = BackTest(open, close, pred['test_pred_rtn'])
        bt_list.append(bt)
    plt.figure(figsize=(16, 6))
    cum_rtn=[]
    for i, bt in enumerate(bt_list):
        cur_pnl = bt.get_cum_rtn()
        cum_rtn.append(cur_pnl)
        cur_pnl.name = method_list[i]
        cur_pnl.plot()
    plt.legend()
    plt.title('PnL Curve')
    plt.show()
    return cum_rtn
