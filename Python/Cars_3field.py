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

data = pd.read_csv("D:/DataCar.csv", header=None,sep=';', index_col=False, names=['line', 'speed', 'gap'])
print("Данные: \n",data.head())
print("Уникальные значения полос: \n", data.line.value_counts())

print("Исходные признаки:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Признаки после get_dummies:\n", list(data_dummies.columns))
print(data_dummies.head())

features = data_dummies
# Извлекаем массивы NumPy
X = features.values
print("форма массива X: {}".format(X.shape))

dataset1 = X
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=12)
kmeans.fit(dataset1)
points = kmeans.labels_
print("Принадлежность к кластерам:\n{}".format(points))
from collections import Counter
c = Counter(points)
print("Всего точек:\n{}".format(points.size))
print("Принадлежность к кластерам:\n{}".format(c))
mglearn.discrete_scatter(dataset1[:, 0], dataset1[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter( kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                          [0, 1, 2,3,4,5,6,7,8,9,10,11], markers='^', markeredgewidth=5)
plt.legend(["кластер 1", "кластер 2","кластер 3","кластер 4","кластер 5",
            "кластер 6","кластер 7","кластер 8","кластер 9","кластер 10","11","12"], loc='best')
plt.xlabel("Скорость, км/ч")
plt.ylabel("Дистанция, м")
print("BuildKMeansModel")
#---


X_new = np.hstack([X[:,0], X[:, 1],X[:,2]])
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# визуализируем в 3D
ax = Axes3D(figure, elev=-152, azim=-26)
y = X[:,2]
# сначала размещаем на графике все точки с y == 0, затем с y == 1
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b', cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^', cmap=mglearn.cm2, s=60)
ax.set_xlabel("признак0")
ax.set_ylabel("признак1")
ax.set_zlabel("признак1 ** 2")

