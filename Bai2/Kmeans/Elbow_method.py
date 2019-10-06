 # clustering dataset
# determine k using elbow method

from sklearn.cluster import KMeans
from sklearn import metrics, datasets
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt


data  = datasets.load_wine()

#Creating the data
x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 5, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)


distortions = []
K = range(1,10)
for k in K:
    #Building and fitting the model
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_,'euclidean'),axis=1)) / X.shape[0])


