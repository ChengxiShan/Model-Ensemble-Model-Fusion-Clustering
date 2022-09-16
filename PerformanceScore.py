from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
def performance_score(method,train, cluster_labels): # 加评价指标注释
    if method == 'silhouette':
        return silhouette_score(train, cluster_labels)
    if method == 'davies_bouldin':
        return davies_bouldin_score(train, cluster_labels)
    if method == 'calinski_harabasz':
        return calinski_harabasz_score(train, cluster_labels)