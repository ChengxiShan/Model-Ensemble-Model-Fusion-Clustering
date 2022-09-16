import pandas as pd
import numpy as np
import lossSet as ls
import cal_IR as cr
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from BackTest import BackTest


class LinearFusion:
    def __init__(self, true_series, models_train_rtn, models_pred_rtn=None, models_train_bt_rtn=None,
                 open=None, close=None):
        # 加入计算bt_rtn方法
        # 汇总模型权重变化
        self.train_rtn = models_train_rtn
        self.pred_rtn = models_pred_rtn
        self.true_rtn = true_series
        self.model_num = models_train_rtn.shape[1]
        self.open = open
        self.close = close
        self.na_weight = None
        self.bt_weight = None
        self.er_weight = None
        self.ic_weight = None
        self.ir_weight = None
        self.lss_weight = None
        self.wt_train_rtn = pd.DataFrame()
        if self.pred_rtn is None:
            self.have_pred = False
        else:
            self.have_pred = True
            self.wt_test_rtn = pd.DataFrame()
        if models_train_bt_rtn is None:
            self.train_bt_rtn=self.cal_train_bt_rtn()
        else:
            self.train_bt_rtn=models_train_bt_rtn

    def cal_train_bt_rtn(self):
        models_nm = self.train_rtn.columns.values
        for i in range(self.model_num):
            cur_model_exp_rtn = self.train_rtn[models_nm[i]]
            try:
                cur_model_bt_rtn = BackTest(self.open, self.close, cur_model_exp_rtn).get_rtn_series()
                return cur_model_bt_rtn
            except:
                print('no open or close data to calculate backtest return!')

    def models_corr(self, set_nm):
        if set_nm == 'train':
            corr_df = self.train_rtn.corr()
        elif self.have_pred:
            corr_df = self.pred_rtn.corr()
        else:
            print('No test set data to calculate correlation!')
            return
        return corr_df

    def naive_avg_weight(self):
        self.na_weight = [1 / self.model_num] * self.model_num
        return self.na_weight

    def bt_rtn_weight(self):
        rtn_mean_list = []
        for model_nm in self.train_rtn.columns:
            ts = self.train_bt_rtn[model_nm]
            try:
                ts.index=pd.to_datetime(ts.index)
            except:
                pass
            dt=self.train_rtn.reset_index().datetime.unique()
            dt=list(set(dt).intersection(set(ts.index.values)))
            ts=ts.loc[dt]
            rtn_mean_list.append(np.mean(ts).values[0])
        rtn_mean_min = min(rtn_mean_list)
        rtn_mean_sum = sum([rtn_mean - rtn_mean_min for rtn_mean in rtn_mean_list])
        self.bt_weight = [(rtn_mean - rtn_mean_min) / rtn_mean_sum for rtn_mean in rtn_mean_list]
        return self.bt_weight

    def error_norm_weight(self, loss_func='mape'):
        model_loss_list = []
        train_df = pd.merge(self.train_rtn, self.true_rtn, right_index=True, left_index=True, how='inner')
        for i in range(self.model_num):
            if loss_func == 'mae':
                loss = ls.mae(train_df.iloc[:, i], train_df.iloc[:, -1])  # val
            elif loss_func == 'mape':
                loss = ls.mape(train_df.iloc[:, i], train_df.iloc[:, -1])
            elif loss_func == 'mse':
                loss = ls.mse(train_df.iloc[:, i], train_df.iloc[:, -1])
            else:
                loss = ls.smape(train_df.iloc[:, i], train_df.iloc[:, -1])
            model_loss_list.append(loss)  # 一维list
        # weight_i=(loss_max-loss_i)/sum_i(loss_max-loss_i)
        loss_max = max(model_loss_list)
        loss_sum = sum([loss_max - loss for loss in model_loss_list])

        self.er_weight = [(loss_max - loss) / loss_sum for loss in model_loss_list]
        return self.er_weight

    def ic_norm_weight(self):
        train_df1 = pd.merge(self.train_rtn, self.true_rtn, right_index=True, left_index=True, how='inner')
        model_ic_list = []
        for i in range(self.model_num):
            cur_corr = train_df1.iloc[:, i].corr(train_df1.iloc[:, -1])
            model_ic_list.append(abs(cur_corr))

        corr_min = min(model_ic_list)
        corr_sum = sum([corr - corr_min for corr in model_ic_list])

        self.ic_weight = [(corr - corr_min) / corr_sum for corr in model_ic_list]
        return self.ic_weight

    def ir_norm_weight(self):
        self.true_rtn.name = 'true_res'
        train_df1 = pd.merge(self.train_rtn, self.true_rtn, right_index=True, left_index=True, how='inner')
        ir_list = []
        for i in range(self.model_num):
            cur_ir = cr.cal_ir(train_df1.iloc[:, i], train_df1.iloc[:, -1])
            ir_list.append(cur_ir)

        ir_min = min(ir_list)
        ir_sum = sum(ir - ir_min for ir in ir_list)

        self.ir_weight = [(ir - ir_min) / ir_sum for ir in ir_list]
        return self.ir_weight

    def lasso_weight(self):
        self.true_rtn.name = 'true_res'
        train_df1 = pd.merge(self.train_rtn, self.true_rtn, right_index=True, left_index=True, how='inner')

        reg_X = train_df1.iloc[:, :-1]
        reg_y = train_df1.iloc[:, -1]

        lassocv = LassoCV()
        lassocv.fit(reg_X, reg_y)
        alpha = lassocv.alpha_
        lasso = Lasso(alpha=alpha)
        lasso.fit(reg_X, reg_y)
        lasso_coefs = list(lasso.coef_)

        lasso_min = min(lasso_coefs)
        lasso_sum = sum([lss - lasso_min for lss in lasso_coefs])

        self.lss_weight = [(lss - lasso_min) / lasso_sum for lss in lasso_coefs]
        return self.lss_weight

    def get_pred(self, set, weight_list):  # train: set=1 test: set=0
        if set:
            pred = self.train_rtn.copy()
            for i in range(self.model_num):
                pred.iloc[:, i] = self.train_rtn.iloc[:, i] * weight_list[i]
            return pred.sum(axis=1)
        elif self.have_pred:
            pred = self.pred_rtn.copy()
            for i in range(self.model_num):
                pred.iloc[:, i] = self.pred_rtn.iloc[:, i] * weight_list[i]
            return pred.sum(axis=1)
        else:
            print('No test set data!')
            return

    def na_train_pred(self):
        self.naive_avg_weight()
        return self.train_rtn.mean(axis=1)

    def bt_train_pred(self):
        self.bt_rtn_weight()
        return self.get_pred(1, self.bt_weight)

    def ew_train_pred(self):
        self.error_norm_weight()
        cur_pred = self.get_pred(1, self.er_weight)
        return cur_pred

    def ic_train_pred(self):
        self.ic_norm_weight()
        cur_pred = self.get_pred(1, self.ic_weight)
        return cur_pred

    def ir_train_pred(self):
        self.ir_norm_weight()
        cur_pred = self.get_pred(1, self.ir_weight)
        return cur_pred

    def lasso_train_pred(self):
        self.lasso_weight()
        cur_pred = self.get_pred(1, self.lss_weight)
        return cur_pred

    def na_test_pred(self):
        self.naive_avg_weight()
        return self.pred_rtn.mean(axis=1)

    def bt_test_pred(self):
        self.bt_rtn_weight()
        return self.get_pred(0, self.bt_weight)

    def ew_test_pred(self):
        self.error_norm_weight()
        cur_pred = self.get_pred(0, self.er_weight)
        return cur_pred

    def ic_test_pred(self):
        self.ic_norm_weight()
        cur_pred = self.get_pred(0, self.ic_weight)
        return cur_pred

    def ir_test_pred(self):
        self.ir_norm_weight()
        cur_pred = self.get_pred(0, self.ir_weight)
        return cur_pred

    def lasso_test_pred(self):
        self.lasso_weight()
        cur_pred = self.get_pred(0, self.lss_weight)
        return cur_pred
