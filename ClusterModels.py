from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from PerformanceScore import performance_score
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import seaborn as sns
import datetime


class K_Means:
    def __init__(self, fea_df, pred_df, n_clusters, random_state,
                 range_n_clusters=None, range_random_state=None,
                 param_select_method='silhouette', select_print=False):
        self.train = fea_df
        self.pred = pred_df
        self.random_state = random_state
        self.range_n_clusters = range_n_clusters
        self.range_random_state = range_random_state
        self.param_select_method = param_select_method
        self.kms = None
        self.train_time_idx = self.train.index
        self.pred_time_idx = self.pred.index
        self.select_print = select_print
        self.n_clusters = n_clusters

    def fit_clusterer(self, n_clusters, random_i, score_list, param_combination_list):
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_i)
        cluster_labels = clusterer.fit_predict(self.train)
        score_nm = self.param_select_method
        score = performance_score(score_nm, self.train, cluster_labels)
        score_list.append(score)
        param_combination_list.append((n_clusters, random_i))
        if self.select_print:
            print(f'For n_clusters={n_clusters} and random_state={random_i}, '
                  f'{self.param_select_method} score is {score}')

    def select_params(self):  # grid search
        if self.n_clusters is not None:
            self.range_n_clusters = self.n_clusters
        elif self.range_n_clusters is None:
            self.range_n_clusters = [2, 3, 4, 5, 6]
        if self.random_state is not None:
            self.range_random_state = self.random_state
        elif self.range_random_state is None:
            self.range_random_state = range(10)

        score_list = []
        param_combination_list = []
        if self.select_print:
            print('')
            print('============================================================================')
        if isinstance(self.range_n_clusters, int):
            n_clusters = self.n_clusters
            for random_i in self.range_random_state:
                self.fit_clusterer(n_clusters=n_clusters, random_i=random_i,
                                   score_list=score_list, param_combination_list=param_combination_list)
        elif isinstance(self.range_random_state, int):
            random_i = self.range_random_state
            for n_clusters in self.range_n_clusters:
                self.fit_clusterer(n_clusters=n_clusters, random_i=random_i,
                                   score_list=score_list, param_combination_list=param_combination_list)
        else:
            for n_clusters in self.range_n_clusters:
                for random_i in self.range_random_state:
                    self.fit_clusterer(n_clusters=n_clusters, random_i=random_i,
                                       score_list=score_list, param_combination_list=param_combination_list)
        score_dic = dict(zip(param_combination_list, score_list))
        max_id = max(score_dic, key=score_dic.get)
        self.n_clusters, self.random_state = max_id
        if self.select_print:
            print('----------------------------------------------------------------------------')
            print('Current best parameters combination is:',
                  'n_clusters=',
                  self.n_clusters,
                  'random_state=',
                  self.random_state,
                  )
            print(f'The {self.param_select_method} score is:',
                  score_dic[max_id])
            print('============================================================================')
            print('')
        return

    def evaluate_model(self):
        if self.n_clusters < 2:
            return

        fig = plt.figure(figsize=(16, 6))
        ax1 = fig.add_subplot(121)

        # The plot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])

        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(self.train) + (self.n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = self.kms
        train_labels = clusterer.labels_
        pred_labels = clusterer.predict(self.pred)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        train_silhouette_avg = silhouette_score(self.train, train_labels)
        train_davies_bouldin_avg = davies_bouldin_score(self.train, train_labels)
        train_calinski_harabasz_avg = calinski_harabasz_score(self.train, train_labels)
        print(
            "For n_clusters =",
            self.n_clusters,
            "random_state=",
            self.random_state,
        )
        print(
            "The average in-sample silhouette_score is :",
            train_silhouette_avg,
        )
        print(
            "The average in-sample davies_bouldin_score is :",
            train_davies_bouldin_avg,
        )
        print(
            "The average in-sample calinski_harabasz_score is :",
            train_calinski_harabasz_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(self.train, train_labels)

        y_lower1 = 10

        for i in range(self.n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[train_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper1 = y_lower1 + size_cluster_i

            color = cm.nipy_spectral(float(i) / self.n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower1, y_upper1),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower1 + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower1 = y_upper1 + 10  # 10 for the 0 samples

            ax1.set_title("The in-sample silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=train_silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

            plt.suptitle(
                f"Silhouette analysis for KMeans clustering on sample data with "
                f"n_clusters = {self.n_clusters}, random_state= {self.random_state}",
                fontsize=12,
                fontweight="bold",
            )
        feature_dimension = len(self.train.columns)
        data = pd.concat([self.train, self.get_train_labels()], axis=1)
        if feature_dimension == 2:
            ax2 = plt.subplot(122)
            data.columns = ['feature1', 'feature2', 'labels']
            sns.scatterplot(x='feature1', y='feature2', hue='labels', data=data)
        if feature_dimension == 3:
            # 绘图
            ax2 = plt.subplot(122, projection='3d')
            ax2.scatter(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2], c=data.iloc[:, 3])
            # 添加坐标轴
            ax2.set_xlabel('feature_0', fontdict={'size': 10, 'color': 'black'})
            ax2.set_ylabel('feature_1', fontdict={'size': 10, 'color': 'black'})
            ax2.set_zlabel('feature_2', fontdict={'size': 10, 'color': 'black'})

        # 加入weight变化图 8条lines
        plt.show()
        timestamp = datetime.datetime.now()
        plt.savefig('clustering evaluation' + str(timestamp) + '.png')

        return

    def train_model(self):
        if (self.n_clusters is None) | (self.random_state is None):
            self.select_params()
        self.kms = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit(self.train)
        return

    def get_train_labels(self):
        lbs = pd.Series(self.kms.labels_)
        lbs.index = self.train_time_idx
        return lbs

    def get_pred_labels(self):
        lbs = self.kms.predict(self.pred)
        lbs = pd.Series(lbs)
        lbs.index = self.pred_time_idx
        return lbs


class Hierarchy:
    def __init__(self,train_df,pred_df,metric='euclidean', linkage_method='ward'):
        self.train=train_df
        self.pred=pred_df
        self.metric = metric
        self.method = linkage_method
        self.Z = None
        self.model=None

    def get_distMat(self,metric=None):  # 生成点与点之间的距离矩阵,这里初始化默认用的欧氏距离
        self.metric = metric if metric is not None else self.metric
        return sch.distance.pdist(self.train, self.metric)

    def hierarchical_division(self,metric=None, method=None):  # 进行层次划分
        self.metric = metric if metric is not None else self.metric
        self.method = method if method is not None else self.method
        distMat = self.get_distMat()
        self.Z = sch.linkage(distMat, method=self.method)

    def plot_division(self):  # 将划分结果以树图保存为plot_dendrogram+datetime.png
        P = sch.dendrogram(self.Z)
        timestamp = datetime.datetime.now()
        plt.savefig('plot_dendrogram_' + str(timestamp) + '.png')

    def train_model(self,distance_threshold=None,n_clusters=None,metric=None, linkage_method=None): # 根据linkage matrix Z得到聚类结果:
        self.metric = metric if metric is not None else self.metric
        self.method = linkage_method if linkage_method is not None else self.method
        self.hierarchical_division()
        self.model = AgglomerativeClustering(n_clusters=n_clusters,affinity=self.metric,linkage=linkage_method,
                                             distance_threshold=distance_threshold).fit(self.train)

    def get_train_labels(self):
        lbs=pd.Series(self.model.labels_)
        lbs.index=self.train.index
        return lbs

    def get_pred_labes(self):
        pass

