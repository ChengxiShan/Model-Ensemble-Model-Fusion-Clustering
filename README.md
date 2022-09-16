# Model-Ensemble-Model-Fusion-Clustering
Ensemble models by applying linear methods based different  indicators like IC/IR/MSE/BackTestReturn. Optimize the ensemble by applying clustering algorithm. Compare the ensemble performance with equal-weighted benchmark performance and show visualizations.

# Model-Fusion-based-on-Clustering-Algorithm

## 功能介绍
* 两大核心模块：线性融合算法与聚类算法，结合实现多个预测模型的融合，提高整体预测效果
* 请参考主程序Test.py为一个使用的Example



## 主程序Test.py

### 导入数据
* 用于融合的各模型训练集与测试集的输出train_data,test_data,若多个periods，则以周期为分割传入list
* 真实回报数据true_rtn
* 日频行情数据prices

### 划分训练集/验证集/测试集
* 确定划分的阶段n_splits
* 根据8:1:1的划分比例大致估算出每阶段上测试集的长度test_size
* 得到各个阶段的训练集this_all_train、验证集this_all_valid、测试集this_all_test
* this_all_XXX是长度为n_splits的list

### 初始化聚类算法
* **models_rtn_train**：**必传参数**，数据类型df，multi-index，各模型训练集回报数据
* **models_rtn_valid**：**必传参数**，数据类型df，multi-index，各模型验证集回报数据
* **models_rtn_test**：**必传参数**，数据类型df，multi-index，各模型测试集回报数据
* **true_rtn**：**必传参数**，数据类型series，multi-index，真实回报
* **model_num**：**必传参数**，数据类型int，待融合模型数量
* **pred_nm**：**必传参数**，数据类型str，预测集名字

        e.g. pred_nm=‘test’
* **n_clusters**：可传参数，数据类型int，默认为None，模型会根据调参方式选择最优n_clusters
* **random_state**：可传参数，数据类型int，默认为None，模型会根据调参方式选择最优random_state
* **weight_method**：可传参数，数据类型str，默认为'ic'，用以选择模型融合方式
        
        其他可选：'ir'/'er'/'lasso'/'na'
* **range_n_clusters**：可传参数，数据类型可以是int或int list，n_clusters的调参范围
        
        若外部不传入则默认为[2, 3, 4, 5, 6]
* **range_random_state**：可传参数，数据类型可以是int或int list，random_state的调参范围

        若外部不传入则默认为range(10)

### 聚类模型训练
* 直接调用**ClusterOptimizer的train_model模块**即可
#### 可选参数
* 训练方式train_method，默认为'KMeans'
* 最优化调参时参考的指标param_select_method，默认为'silhouette'
* 调参过程是否输出verbose，默认为True

### 聚类模型评价与可视化
#### 评价指标
##### Silhouette Coefficient
* Silhouette Coefficient=(b-a)/max(a,b)
* a=样本到同簇点的平均距离
* b=样本到不同簇点的平均距离
* 指标越接近1，聚类越合理

##### Calinski Harabasz Score
* 定义为簇间离散与簇内离散的比率，是通过评估类之间方差和类内方差来计算得分
* 该分值越大说明聚类效果越好

##### Davies Bouldin Score
* 计算任意两类别的类内距离平均距离(CP)之和除以两聚类中心距离求最大值
* 该分值越小意味着类内距离越小同时类间距离越大

### 线性融合
#### 获得每天各模型选取的权重
* 直接调用**ClusterOptimizer**的**get_weightl模块**即可
#### 获得训练集每天的预测结果
* 直接调用**ClusterOptimizer**的**get_pred模块**即可

### 输出保存
* 模型权重
* 调参过程
* 模型评价可视化结果

## 文件Linear Fusion.py

### 功能
#### 1. 查看待融合的模型预测结果的相关性
* 调用models_corr函数
* 传入参数set_nm:

        e.g.LinearFusion.models_corr(set_nm='train')
        'train'表示查看训练集上的相关性

#### 2. 实现等权/误差归一化/ic归一化/ir归一化/lasso归一化等五种线性融合方法
##### 1）等权
* naive_avg_weight返回模型权重
* na_train_pred返回训练集融合预测结果
* na_test_pred返回测试集融合预测结果

##### 2）误差归一化
* error_norm_weight返回模型权重，loss_func参数可选误差函数'mse’,'mae','mape'
* ew_train_pred返回训练集融合预测结果
* ew_test_pred返回测试集融合预测结果

##### 3）ic归一化
* ic_norm_weight返回模型权重
* ic_train_pred返回训练集融合预测结果
* ic_test_pred返回测试集融合预测结果

##### 4）ir归一化
* ir_norm_weight返回模型权重
* ir_train_pred返回训练集融合预测结果
* ir_test_pred返回测试集融合预测结果

##### 5) lasso归一化
* lasso_weight返回模型权重
* lasso_train_pred返回训练集融合预测结果
* lasso_test_pred返回测试集融合预测结果
* 
##### 5) backtest_rtn归一化
* bt_rtn_weight返回模型权重
* bt_rtn_train_pred返回训练集融合预测结果
* bt_rtn_test_pred返回测试集融合预测结果

### 初始化init
* **true_series:**：真实的y_labels，数据类型series
* **models_train_rtn:**：待融合的各模型的训练集预测，数据类型df
* **models_pred_rtn:**：待融合的各模型的预测集预测，数据类型df


## 文件GetGroupWeight.py
### 功能
* 获得每个簇中各模型对应的权重
### input
* train_labels:聚类模型训练集中每天对应的簇labels，数据类型series
* pred_labels:聚类模型预测集中每天对应的簇labels，数据类型series
* train_df:各模型的训练集数据，数据类型df
* pred_df:各模型的预测集数据，数据类型df
* true_rtn:真实的rtn，数据类型series
* weight_method:线性加权方法，默认为'ic'，数据类型str

        其他可选'ir'/'na'/'lasso'/'er'

### output
* train_wt_df:模型训练集中每天对应的模型权重，数据类型df
* pred_wt_df:模型预测集中每天对应的模型权重，数据类型df

