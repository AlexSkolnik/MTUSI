import sys
print("версия Python: {}".format(sys.version))
import pandas as pd
print("версия pandas: {}".format(pd.__version__))
import numpy as np
print("версия NumPy: {}".format(np.__version__))
import scipy as sp
print("версия SciPy: {}".format(sp.__version__))
import IPython
print("версия IPython: {}".format(IPython.__version__))
import sklearn
print("версия scikit-learn: {}".format(sklearn.__version__))
import mglearn
print("версия mglearn: {}".format(mglearn.__version__))
import matplotlib
print("версия matplotlib: {}".format(matplotlib.__version__))
import matplotlib.pyplot as plt
import csv
print("версия csv: {}".format(csv.__version__))
from sklearn.datasets import make_blobs
from sklearn.metrics.cluster import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from collections import Counter

dataset = np.loadtxt("D:/Traffic.csv", delimiter=";")
X = dataset[:, 0]
y = dataset[:, 1]

# two = dataset[:,:2]
# scaler = StandardScaler()
# #scaler = MinMaxScaler()
# scaler.fit(two)
# X_scaled = scaler.transform(two)
#
# kmeans = KMeans(n_clusters=6)
# kmeans.fit(X_scaled)
# points = kmeans.labels_
# print("Принадлежность к кластерам:\n{}".format(points))
# c = Counter(points)
# print("Всего точек:\n{}".format(points.size))
# print("Принадлежность к кластерам:\n{}".format(c))
# mglearn.discrete_scatter(X_scaled[:, 0], X_scaled[:, 1], kmeans.labels_, markers='o')
# mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
#                          [0,1, 2, 3, 4, 5], markers='^', markeredgewidth=3)
# plt.legend(["кластер 1", "кластер 2", "кластер 3", "кластер 4", "кластер 5",
#             "кластер 6"], loc='best')
# plt.xlabel("label_id")
# plt.ylabel("flow_id")


#
# agg = AgglomerativeClustering(n_clusters=6)
# assignment = agg.fit_predict(X_scaled)
# mglearn.discrete_scatter(X_scaled[:, 0], X_scaled[:, 1], assignment)
# plt.legend(["кластер 1", "кластер 2", "кластер 3", "кластер 4", "кластер 5","кластер 6"], loc='best')
# plt.xlabel("label_id")
# plt.ylabel("flow_id")
# print("BuildAgglomerativeClustering")


two = dataset[:,:2]
#scaler = StandardScaler()
scaler = MinMaxScaler()
scaler. fit(two)
X_scaled = scaler.transform(two)
plt.ylabel("flow_id")
plt.xlabel("label_id")
mglearn.discrete_scatter(X_scaled[:, 0], X_scaled[:, 1])
#mglearn.discrete_scatter(two[:, 0], two[:, 1])


mas = np.empty(shape=[0, 3])
deltaEPS = 0.01;
eps = 0
while eps < 0.1:
    min_samples = 1
    while min_samples < 100:
        eps = eps + deltaEPS;
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        # print("Принадлежность к кластерам:\n{}".format(clusters))
        min = np.min(clusters)
        max = np.max(clusters)
        c = Counter(clusters)
        # print("Принадлежность к кластерам:\n{}".format(c))
        # print("Всего точек:\n{}".format(clusters.size))

        print("Нормальное число кластеров: {}".format(max - min))
        print("eps = {}, min_samples = {}".format(eps,min_samples))
        mas = np.append(mas, [[eps,min_samples,(max - min)]], axis=0)
        min_samples = min_samples + 5;


np.savetxt('D:/otputMas.txt', mas)




