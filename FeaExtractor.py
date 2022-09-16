import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


def extract_corr_feature(raw_train_df, model_num):
    corr_df = raw_train_df.groupby('datetime').corr()
    dt = corr_df.reset_index().datetime.unique()
    fea_df = []
    for d in dt:
        corr = corr_df.loc[d, :].values.tolist()
        cur_fea = []
        for i in range(model_num - 1):
            cur_fea += corr[i][i + 1:model_num]
        fea_df.append(cur_fea)
    fea_df = pd.DataFrame(fea_df)
    fea_df.index=dt
    return fea_df


def extract_expt_rtn_feature(train_df,reduce_dim=True, retained_var=0.9):
    dt = train_df.reset_index().datetime.unique()
    df = train_df.apply(lambda x: x.replace(np.nan, np.mean(x)), axis=0)
    df = df.apply(lambda x: x - np.mean(x), axis=0)
    fea_df = df.unstack(fill_value=0).values
    if reduce_dim: # if pca, 根据方差阈值选择特征数量
        fea_df=pca_reduce_dim(fea_df, retained_var)
    fea_df.index = dt
    return fea_df


def pca_reduce_dim(pri_df, retained_var=0.9):
    n_features = pri_df.shape[1]
    for i in range(2, n_features + 1):
        pca = PCA(n_components=i)
        pca.fit(pri_df)
        var = sum(pca.explained_variance_)
        if var > retained_var:
            transformed_fea_df = pca.fit_transform(pri_df)
            transformed_fea_df = pd.DataFrame(transformed_fea_df)
            return transformed_fea_df
        else:
            continue


