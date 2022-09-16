import trans2df as tdf
import pandas as pd


class BackTest:
    def __init__(self, open_series, close_series, pred_series, num=20, capital=100000000, cost=0.002):
        pred_series.name = 'signal'
        open_sr = open_series.rename_axis(['datetime', 'asset'])
        close_sr = close_series.rename_axis(['datetime', 'asset'])
        pred_series = pred_series.rename_axis(['datetime', 'asset'])
        open_sr = pd.merge(open_sr, pred_series, right_index=True, left_index=True, how='inner')['open_price']
        close_sr = pd.merge(close_sr, pred_series, right_index=True, left_index=True, how='inner')['close_price']
        self.open_df = tdf.trans_df(open_sr)
        self.close_df = tdf.trans_df(close_sr)
        self.signal_df = tdf.trans_df(pred_series)
        self.num = num
        self.capital = capital
        self.cost = cost
        self.long_pos = None
        self.short_pos = None

    def get_long_pos(self):
        long = (self.signal_df.rank(axis=1, ascending=False)) <= 20
        long = long.shift(1) # 隔日信号对应第二天的仓位
        self.long_pos = ((self.capital / (2 * self.num)) / self.open_df) * long
        return

    def get_short_pos(self):
        short = (self.signal_df.rank(axis=1)) <= 20
        short = short.shift(1)
        self.short_pos = ((self.capital / (2 * self.num)) / self.open_df) * short
        return

    def get_rtn_series(self):
        self.get_long_pos()
        self.get_short_pos()
        long_settle = self.long_pos * (self.close_df - self.open_df) * (1 - self.cost)
        short_settle = self.short_pos * (self.open_df - self.close_df) * (1 - self.cost)
        rtn_series = (long_settle.sum(axis=1) + short_settle.sum(axis=1)) / self.capital
        return rtn_series

    def get_cum_rtn(self):
        rtn_sr = self.get_rtn_series()
        cum_rtn = (1 + rtn_sr).cumprod()
        return cum_rtn

    def plot_pnl(self):
        cum_rtn = self.get_cum_rtn()
        cum_rtn.plot()
