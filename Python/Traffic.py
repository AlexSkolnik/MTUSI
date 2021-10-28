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


df = pd.read_csv("D:/Traffic.csv",header=None, sep=';')
print(df)
print("форма массива df: {}".format(df.shape))

X1 = df[: 0]
print("Массив Х1: {}",X1)
print("форма массива X1: {}".format(X1.shape))


X2 = df[: 0]
print("Массив Х2: {}",X2)
print("форма массива X2: {}".format(X2.shape))

Y = df[1]
print("Массив Y: {}",Y)
print("форма массива Y: {}".format(Y.shape))

# генерируем синтетические двумерные данные строим модель кластеризации


kmeans = KMeans(n_clusters=6)
kmeans.fit(df)
print("Принадлежность к кластерам:\n{}".format(kmeans.labels_))
print(kmeans.predict(df))

mglearn.discrete_scatter(df[0], df[1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],[0, 1, 2,3,4,5], markers='^', markeredgewidth=2)
plt.legend(["кластер 1", "кластер 2","кластер 3","кластер 4","кластер 5", "кластер 6",], loc='best')
plt.close()



###
# with open(csv_path, "r") as f_obj:
#     reader = csv.reader(file_obj)
#     str = []
#     for row in reader:
#         print(" ".join(row))
#         str += row;
dataset = np.loadtxt("D:/Traffic.csv", delimiter=";")
X = dataset[:, 0]
y = dataset[:, 1]
print("X.shape:", X.shape)
print("y.shape:", y.shape)
plt.plot(X, y, 'o')
plt.close()

scaler = StandardScaler()
#scaler = MinMaxScaler()
scaler.fit(dataset)
X_scaled = scaler.transform(dataset)

dbscan = DBSCAN(eps=0.005, min_samples=30)
clusters = dbscan.fit_predict(X_scaled)
print("Принадлежность к кластерам:\n{}".format(clusters))
# выводим принадлежность к кластерам
plt.show()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
plt.xlabel("LabelId")
plt.ylabel("FlowId")

plt.close()






X = dataset[:,:2]
# масштабируем данные так, чтобы получить нулевое среднее и единичную дисперсию
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
fig, axes = plt.subplots(1, 4, figsize=(15, 3),subplot_kw={'xticks': (), 'yticks': ()})

# случайно присваиваем точки двум кластерам для сравнения
random_state = np.random.RandomState(seed=0)
random_clusters = random_state.randint(low=0, high=2, size=len(X))
# выводим на графике результаты случайного присвоения кластеров
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters,cmap=mglearn.cm3, s=60)
axes[0].set_title("Случайное присвоение кластеров: {:.2f}".format(silhouette_score(X_scaled, random_clusters)))


algorithms = [KMeans(n_clusters=6), AgglomerativeClustering(n_clusters=6),DBSCAN()]
for ax, algorithm in zip(axes[1:], algorithms):
    clusters = algorithm.fit_predict(X_scaled)
    # выводим на графике принадлежность к кластерам и центры кластеров
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm3,s=60)
    ax.set_title("{} : {:.2f}".format(algorithm.__class__.__name__,
    silhouette_score(X_scaled, clusters)))
plt.show()
plt.show()