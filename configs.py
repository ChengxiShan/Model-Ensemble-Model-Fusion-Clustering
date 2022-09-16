import pandas as pd

# path relevant
n_periods = 6  # 训练周期数
true_rtn_path = 'new_data/true_return.pkl'  # 实际收益序列
prices_path = 'data/prices.csv'
# --------------------------------------train------------------------------------------
train_pre_path = 'new_data/training_data/'  # model_x文件夹之前的公共路径
train_pre_nm = 'training_return_series_'  # model_x文件夹内文件名的共同前缀
models_nm_list = ['model_a', 'model_c', 'model_l', 'model_g']  # models name
# 每个period training开始的日期
start_data_list = ['20180301', '20180601', '20180901', '20181201', '20190301', '20190601']
# 每个period training结束的日期
end_data_list = ['20210228', '20210531', '20210831', '20211130', '20220228', '20220531']
# --------------------------------------test------------------------------------------
test_pre_path = 'new_data/test_data/'  # model_x文件夹之前的公共路径
test_pre_nm = 'factor_return_'  # model_x文件夹内文件名的共同前缀
# 每个period test开始的年份
start_y_list = [2021, 2021, 2021, 2022, 2022, 2022]
# 每个period test结束的年份
end_y_list = [2021, 2021, 2021, 2022, 2022, 2022]
# 每个period test开始的月份
start_m_list = [4, 7, 10, 1, 4, 7]
# 每个period test结束的月份
end_m_list = [6, 9, 12, 3, 6, 9]
# ----------------------------models-backtest-rtn-series-on-train-set--------------------
# 各模型的回测收益率
bt_rtn_pre_path = ''
bt_rtn_list = []
for i in range(6):
    period_rtn = []
    for model_nm in models_nm_list:
        bt_rtn = pd.read_csv(bt_rtn_pre_path + 'bt_rtn_period' + str(i) + '_' + model_nm + '.csv')
        bt_rtn.columns = ['datetime', 'expected_rtn']
        bt_rtn = bt_rtn.set_index('datetime')
        period_rtn.append(bt_rtn)
    period_rtn = dict(zip(models_nm_list, period_rtn))
    bt_rtn_list.append(period_rtn)

# ----------------------------Linear-Fusion-Methods-Comparison----------------------------
compare_methods_list = ['bt_rtn', 'na', 'ic', 'er']  # methods to compare

# ----------------------------Clustering-based-on-Linear-Fusion---------------------------
n_clusters_list = [None] * n_periods  # 每个阶段规定聚类为几类，不规定则设置为None，将自动调参
random_state_list = [None] * n_periods  # 每个阶段规定初始化state，不规定则设置为None，将自动调参
range_n_clusters = None  # 每次聚类n_clusters的调餐范围
range_random_state = None  # 每次聚类random_state的调餐范围
cluster_size = 20  # 辅助参数，期望类的大小
model_num = 4  # 模型个数
clustering_train_method = 'KMeans'  # 设置聚类方式
param_select_method = 'silhouette'  # 设置用于挑选参数的评价指标
# {'silhouette','davies_bouldin','calinski_harabasz'}

fea_nm = 'corr'  # 选择的特征 {'all','corr'}
verbose = True  # 是否输出调餐过程
gnl = True  # 是否多次平均提升泛化
if_reduce_dim = True  # 是否pca降维处理特征
retained_var = 0.9  # 降维后需要保留的方差百分比
lf_weight_method = 'bt_rtn'  # 聚类分组后每组使用的加权方法
# {'ic','ir','er','na','lasso','bt_rtn'}

# ---------------Compare-Clustering-and-Non-clustering-Linear-Fusion-PnL--------------------
benchmark_method = 'bt_rtn'  # {'ic','ir','er','na','lasso','bt_rtn'}
pred_method_list = ['cluster_bt_rtn', 'bt_rtn']