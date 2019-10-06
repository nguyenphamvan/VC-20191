from __future__ import print_function
import numpy as np
from sklearn.datasets import make_blobs
from sklearn import metrics, datasets
from sklearn.metrics.cluster import completeness_score
np.random.seed(11)

means = [[2, 2], [8, 3], [3, 6]]
cov = [[1, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

data = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


# Tạo dữ liệu
# data, label= make_blobs(n_samples=150,         # tổng số điểm sample
#                         n_features=2,          # số lượng feature(số chiều) default:2
#                         centers=3,             # số lượng cluster
#                         cluster_std=0.5,       # độ lệch chuẩn giữa các cluster
#                         shuffle=True,
#                         random_state=0)        #  có trộn các sample với nhau không

# iris = datasets.load_iris()
# data = iris.data
# label = iris.target

# K = 3

def init_centers(X, k):
    return X[np.random.choice(X.shape[0], k, replace=False)]

def assign_labels(X, centers):
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        d = X[i] - centers
        d = np.linalg.norm(d, axis=1)
        y[i] = np.argmin(d)

    return y

def update_centers(X, labels, K):
    centers = np.zeros((K, X.shape[1]))
    for i in range(K):
        X_i = X[labels == i, :]
        centers[i,:] = np.mean(X_i, axis=0)

    return centers

def has_converged(centers, new_centers):
    return (set([tuple(a) for a in centers]) ==
        set([tuple(a) for a in new_centers]))

def kmeans(X, K):
    centers = [init_centers(X, K)]
    labels = []
    it = 0
    while True:
        labels.append(assign_labels(X, centers[-1]))
        new_centers = update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels[-1], it)

(centers, pred_label_no, it) = kmeans(data, K)
print('Thuật toán dừng sau số bước lặp = ',it)
print('Các tâm cụm tìm bởi thuật toán tự cài đặt với len(data) = ',len(data))
print(centers[-1])
print()

# Sử dụng thư viện KMeans trong sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,
                init='random',
                n_init=10,
                max_iter=300,
                tol=1e-04,
                random_state=0)
kmeans.fit(data)
print('Các tâm cụm tìm bởi scikit-learn với len(data) = ',len(data))
print(kmeans.cluster_centers_)
pred_label = kmeans.predict(data)


print("Test accuracy không dùng thư viện : ",completeness_score(original_label,pred_label_no))
print("Test accuracy dùng thư viện sklearn: ",completeness_score(original_label, pred_label))



