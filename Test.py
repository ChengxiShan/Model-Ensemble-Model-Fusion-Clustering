import pandas as pd
from functools import reduce
import ClusterOptimizer as CO
from LfMethodsComparision import compare_methods
from BackTest import BackTest
import matplotlib.pyplot as plt
from LinearFusion import LinearFusion
import warnings
warnings.filterwarnings("ignore")
from configs import *


# 1.load data
def training_data_loader(pre_path, pre_nm, model_nm_list, start_date_list, end_date_list):
    train_data = []
    for i in range(len(start_date_list)):
        cur_period_data_list = []
        for model_nm in model_nm_list:
            path = pre_path + model_nm + '/' + pre_nm + start_date_list[i] + '_' + end_date_list[i] + '.pkl'
            cur_model = pd.read_pickle(path)
            cur_model.name = model_nm
            cur_period_data_list.append(cur_model)
        cur_period = reduce(lambda x, y: pd.merge(x, y, left_index=True, right_index=True), cur_period_data_list)
        cur_period.columns = models_nm_list
        train_data.append(cur_period)
    return train_data


def to_str(i):
    if i > 10:
        return str(i)
    else:
        return '0' + str(i)


def test_data_loader(pre_path, pre_nm, model_nm, start_y, end_y, start_m, end_m):
    data = pd.DataFrame()
    for y in range(start_y, end_y + 1):
        if start_y == end_y:
            iter_m = range(start_m, end_m + 1)
        elif y == start_y:
            iter_m = range(start_m, 13)
        elif y == end_y:
            iter_m = range(1, end_m + 1)
        else:
            iter_m = range(1, 13)
        y = to_str(y)
        for m in iter_m:
            m = to_str(m)
            for d in range(1, 32):
                d = to_str(d)
                nm = y + m + d
                path = pre_path + model_nm + '/' + pre_nm + nm + '_' + nm + '.csv'
                try:
                    daily_pred = pd.read_csv(path)
                    daily_pred.datetime = pd.to_datetime(daily_pred.datetime)
                    daily_pred.asset = daily_pred.asset.apply(lambda x: str(x))
                    data = pd.concat([data, daily_pred], axis=0)
                except:
                    continue
    return data.set_index(['datetime', 'asset'])


def models_test_exp_rtn(pre_path, pre_nm, models_nm_list, start_y_list, end_y_list, start_m_list, end_m_list):
    n = len(start_m_list)
    exp_rtn_list = []
    for i in range(n):
        models = []
        for model_nm in models_nm_list:
            models.append(test_data_loader(pre_path, pre_nm, model_nm,
                                           start_y_list[i], end_y_list[i],
                                           start_m_list[i], end_m_list[i]))
        models_test_expected_rtn = pd.concat(models, axis=1, join='inner')
        models_test_expected_rtn.columns = models_nm_list
        exp_rtn_list.append(models_test_expected_rtn)
    return exp_rtn_list


# 1.1 true res
true_rtn = pd.read_pickle(true_rtn_path)  # true_rtn_path

# 1.2 daily market data
prices = pd.read_csv(prices_path)
prices.trade_date = pd.to_datetime(prices.trade_date)
prices = prices.set_index(['trade_date', 'symbol_id'])
open = prices.open_price
close = prices.close_price

# 1.3 set parameters
train_data = training_data_loader(train_pre_path, train_pre_nm, models_nm_list, start_data_list, end_data_list)
test_data = models_test_exp_rtn(test_pre_path, test_pre_nm, models_nm_list, start_y_list, end_y_list, start_m_list,
                                end_m_list)

# 2.Linear Fusion methods comparison
compare_methods(train_data, test_data, true_rtn, n_periods, compare_methods_list, open, close, bt_rtn_list)

# 3.Clustering based on Linear Fusion
# 3.1 initiate all periods
op_list = []
for i in range(n_periods):
    op = CO.ClusterOptimizer(models_rtn_train=train_data[i], models_rtn_test=test_data[i], true_rtn=true_rtn,
                             model_num=model_num, pred_nm='test', n_clusters=n_clusters_list[i],
                             random_state=random_state_list[i], range_n_clusters=range_n_clusters,
                             range_random_state=range_random_state)
    op_list.append(op)

# 3.2 Train model
# set parameters
for op in op_list:
    op.train_model(train_method=clustering_train_method, param_select_method=param_select_method,
                   feature_nm=fea_nm, verbose=verbose, generalize=gnl, reduce_dim=if_reduce_dim,
                   retained_var=retained_var)

# 3.3 Evaluate model parameters and visualization (optional)
# here is an example
# op_list[0].model.evaluate_model()

# 4. Linear Fusion After Clustering
# 4.1 Get weight

i = 0
for op in op_list:
    op.get_weight(weight_method=lf_weight_method, bt_rtn=bt_rtn_list[i], open=open, close=close)
    i += 1

# 4.2 Get weighted rtn
for op in op_list:
    op.get_weighted_rtn(weight_method=lf_weight_method)


# 5.Evaluate Pnl
# 计算benchmark
def get_benchmark_pred(train_data, test_data, true_rtn, n_splits, method, backtest_rtn_list=None):
    pred_list1 = []
    pred_list2 = []

    for i in range(n_splits):
        lf = LinearFusion(true_series=true_rtn, models_train_rtn=train_data[i], models_pred_rtn=test_data[i],
                          models_train_bt_rtn=backtest_rtn_list[i], open=open, close=close)
        if method == 'ic':
            pred_list1.append(lf.ic_test_pred())
            pred_list2.append(lf.ic_train_pred())
        elif method == 'ir':
            pred_list1.append(lf.ir_test_pred())
            pred_list2.append(lf.ir_train_pred())
        elif method == 'er':
            pred_list1.append(lf.ir_test_pred())
            pred_list2.append(lf.ew_train_pred())
        elif method == 'bt_rtn':
            pred_list1.append(lf.bt_test_pred())
            pred_list2.append(lf.bt_train_pred())

    pred_rtn = pd.concat([pred for pred in pred_list1])
    train_rtn = pd.concat([pred for pred in pred_list2])

    benchmark_return_items = [train_rtn, pred_rtn]
    benchmark_return_id = ['train_pred_rtn', 'test_pred_rtn']
    benchmark_return = dict(zip(benchmark_return_id, benchmark_return_items))
    return benchmark_return


# 5.1 Aggregate all clustering model pred
cluster_pred_rtn = pd.concat([op.new_pred_rtn for op in op_list], axis=0)
benchmark_rtn = get_benchmark_pred(train_data, test_data, true_rtn, n_periods, benchmark_method, bt_rtn_list)
benchmark_test_pred = benchmark_rtn['test_pred_rtn']


# 5.2 Compare Clustering and Non-clustering Linear Fusion PnL
def pnl_compare_plot(open, close, pred_list, pred_method_list):
    n_methods = len(pred_method_list)
    plt.figure(figsize=(16, 6))
    for i in range(n_methods):
        pred_rtn = BackTest(open, close, pred_list[i]).get_cum_rtn()
        pred_rtn.name = pred_method_list[i]
        pred_rtn.plot()
        print(pred_method_list[i])
        print(pred_rtn)
    plt.legend()
    plt.title('PnL Curve')
    plt.show()


pred_list = [cluster_pred_rtn, benchmark_test_pred]
pnl_compare_plot(open, close, pred_list, pred_method_list)
