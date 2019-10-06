from __future__ import print_function
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn import metrics, datasets
from sklearn.metrics.cluster import completeness_score

# create data
# (x,y)
# data, label= make_blobs(n_samples=150,         # tổng số điểm sample
#                         n_features=2,          # số lượng feature(số chiều) default:2
#                         centers=3,             # số lượng cluster
#                         cluster_std=0.5,       # độ lệch chuẩn giữa các cluster
#                         shuffle=True)        #  có trộn các sample với nhau không

iris = datasets.load_iris()
data = iris.data
label = iris.target


kmeans = KMeans(n_clusters=3, random_state=0).fit(data)
print('Centers found by scikit-learn:')
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(data)
print(pred_label)
print("Test accuracy: ",completeness_score(pred_label, label))
