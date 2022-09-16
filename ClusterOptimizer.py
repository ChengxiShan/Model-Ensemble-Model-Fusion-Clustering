import pandas as pd
import datetime
from functools import reduce
import FeaExtractor as fetc
import GetGroupWeight as ggw
import ClusterModels as CM


class ClusterOptimizer:

    def __init__(self, models_rtn_train,models_rtn_test,true_rtn, model_num,
                 pred_nm, models_rtn_valid=None, n_clusters=None, random_state=None, range_n_clusters=None,
                 range_random_state=None):

        self.train = models_rtn_train # all models expected rtn for train set
        self.pred_nm = pred_nm
        self.model_num = model_num
        self.true_res = true_rtn
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.range_n_clusters = range_n_clusters
        self.range_random_state = range_random_state
        self.train_labels = []
        self.pred_labels = []
        self.train_fea = None
        self.pred_fea = None
        self.train_weight_df = None
        self.pred_weight_df = None
        self.new_train_rtn = None
        self.new_pred_rtn = None
        self.model = None
        self.pred = models_rtn_valid if self.pred_nm == 'valid' else models_rtn_test

    def extract_fea(self, feature_nm, reduce_dim,retained_var):  # 特征提取
        if feature_nm == 'corr':
            train_fea_df = fetc.extract_corr_feature(self.train, self.model_num)
            test_fea_df = fetc.extract_corr_feature(self.pred, self.model_num)
        elif feature_nm == 'all':
            train_fea_df = fetc.extract_expt_rtn_feature(self.train, reduce_dim,retained_var)
            test_fea_df = fetc.extract_expt_rtn_feature(self.pred, reduce_dim,retained_var)
        else:
            train_fea_df = None
            test_fea_df = None
        return train_fea_df, test_fea_df

    def re_init_params(self, n_clusters=None, random_state=None,
                       range_n_clusters=None, range_random_state=None):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.range_n_clusters = range_n_clusters
        self.range_random_state = range_random_state

    def k_means(self, n_clusters, param_select_method, select_print):
        return CM.K_Means(self.train_fea, self.pred_fea, n_clusters, self.random_state, self.range_n_clusters,
                          self.range_random_state, param_select_method, select_print)

    def train_model(self, train_method='KMeans', param_select_method='silhouette',
                    feature_nm='all', verbose=True, generalize=True, reduce_dim=True,retained_var=0.9):
        # train_method: 选用何种算法训练模型
        # param_select_method: 聚类模型依据何种评价指标选取最优参数
        # verbose=True输出中间的训练过程
        # feature_nm: 提取的特征名字，'all'表示选用所有predicted rtn信息，'corr'表示选用模型间相关性信息
        # retained_var: 对特征降pca维化处理需要保留的方差百分比
        # generalize: 是否需要进行多次平均进行泛化处理
        self.train_fea, self.pred_fea = self.extract_fea(feature_nm=feature_nm,reduce_dim=reduce_dim,retained_var=retained_var)
        if train_method == 'KMeans':
            self.model = self.k_means(self.n_clusters, param_select_method, verbose)
            self.model.train_model()
            self.train_labels.append(self.model.get_train_labels())
            self.pred_labels.append(self.model.get_pred_labels())
            self.n_clusters = self.model.n_clusters
            if generalize:
                if self.n_clusters <= 2:
                    generalize_list = [self.n_clusters + 1, self.n_clusters + 2]
                else:
                    generalize_list = [self.n_clusters - 1, self.n_clusters + 1]
                for n_clusters in generalize_list:
                    sub_model = self.k_means(n_clusters, param_select_method, select_print=False)
                    sub_model.train_model()
                    self.train_labels.append(sub_model.get_train_labels())
                    self.pred_labels.append(sub_model.get_pred_labels())

        return

    def get_weight(self, weight_method,bt_rtn=None,open=None,close=None):
        # 加一下函数功能注释
        """
        this function is built to get daily weight of each model according to the given method

        :param weight_method: apply which linear fusion method to identify model weight
        :return: daily_weight_df for both train and pred data
        """
        if (weight_method=='bt_rtn')&(bt_rtn==None):
            if (open==None)|(close==None):
                print('no open or close data passed')

        train_weight_df_list = []
        pred_weight_df_list = []
        # 加模型数量参数 assert
        weighted_avg_models_num = len(self.train_labels)
        assert weighted_avg_models_num == 1 | weighted_avg_models_num == 3, '加权平均的模型数量不正确'
        for i in range(weighted_avg_models_num):
            train_weight_df, pred_weight_df = ggw.get_group_weight(self.train_labels[i], self.pred_labels[i],
                                                                   self.train, self.pred, self.true_res,
                                                                   weight_method,bt_rtn=bt_rtn,open=open,
                                                                   close=close)
            train_weight_df_list.append(train_weight_df)
            pred_weight_df_list.append(pred_weight_df)

        self.train_weight_df = reduce(lambda x, y: (x + y), train_weight_df_list) / weighted_avg_models_num
        self.pred_weight_df = reduce(lambda x, y: (x + y), pred_weight_df_list) / weighted_avg_models_num

        timestamp = datetime.datetime.now()
        self.train_weight_df.to_csv('train_set_weight' + str(timestamp) + '.csv')
        self.pred_weight_df.to_csv('pred_set_weight' + str(timestamp) + '.csv')
        return

    def get_weighted(self, weight_df, df):
        """
        this function is built to use daily weight df and given models' daily rtn predictions to
        calculate the weighted average rtn
        """
        new_df = pd.DataFrame()
        n = self.model_num
        for index, row in weight_df.iterrows():
            cur_df = df.loc[(index, slice(None)),:]
            cur_new_df = pd.DataFrame()
            for i in range(n):
                tmp = cur_df.iloc[:, i] * row[i]
                cur_new_df = pd.concat([cur_new_df, tmp], axis=1)
            new_df = pd.concat([new_df, cur_new_df], axis=0)
        new_df = new_df.sum(axis=1)
        new_df.index = df.index
        return new_df

    def get_weighted_rtn(self,weight_method='ic'):
        if self.train_weight_df is None:
            self.get_weight(weight_method)
        self.new_train_rtn = self.get_weighted(self.train_weight_df, self.train)
        self.new_pred_rtn = self.get_weighted(self.pred_weight_df, self.pred)
        return
