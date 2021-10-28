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

# ----------------------------- Чтение файла ----------------------------------
def csv_reader(file_obj):
    reader = csv.reader(file_obj)
    str = []
    for row in reader:
        print(" ".join(row))
        str += row;
# ------------------------------------------------------------------------------
# ----------------------------- Извлечениеданных из файла----------------------------------
def GetDataSetFromFile(csv_path):
    with open(csv_path, "r") as f_obj:
        csv_reader(f_obj)
    dataset = np.loadtxt(csv_path, delimiter=";")
    X = dataset[:, 0]
    y = dataset[:, 1]
    print("X.shape:", X.shape)
    print("y.shape:", y.shape)
    plt.plot(X, y, 'o')
    plt.ylim(0, 70)
    plt.xlim(0, 140)
    plt.xlabel("Скорость, км/ч")
    plt.ylabel("Дистанция, м")
    return dataset;
# -----------------------------------------------------------
#------------------------- Модель кластеризации  к-средних --------------------
def BuildKMeansModel(dataset1):
    from sklearn.datasets import make_blobs
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=10)
    kmeans.fit(dataset1)
    points = kmeans.labels_
    print("Принадлежность к кластерам:\n{}".format(points))
    from collections import Counter
    c = Counter(points)
    print("Всего точек:\n{}".format(points.size))
    print("Принадлежность к кластерам:\n{}".format(c))
    mglearn.discrete_scatter(dataset1[:, 0], dataset1[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter( kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
                                [0, 1, 2,3,4,5,6,7,8,9], markers='^', markeredgewidth=5)
    plt.legend(["кластер 1", "кластер 2","кластер 3","кластер 4","кластер 5",
                "кластер 6","кластер 7","кластер 8","кластер 9","кластер 10"], loc='best')
    plt.xlabel("Скорость, км/ч")
    plt.ylabel("Дистанция, м")
    print("BuildKMeansModel")
#----------------------------------------------------------------------------------

#------------------------- Модель кластеризации  Agglomerative --------------------
def BuildAgglomerativeClustering(dataset):
    from sklearn.cluster import AgglomerativeClustering
    agg = AgglomerativeClustering(n_clusters=10)
    assignment = agg.fit_predict(dataset)
    mglearn.discrete_scatter(dataset[:, 0], dataset[:, 1], assignment)
    plt.legend(["кластер 1", "кластер 2", "кластер 3", "кластер 4", "кластер 5",
                "кластер 6", "кластер 7", "кластер 8", "кластер 9", "кластер 10"], loc='best')
    plt.xlabel("Скорость, км/ч")
    plt.ylabel("Дистанция, м")
    print("BuildAgglomerativeClustering")
#---------------------------------------------------------------------

#------------------------- Модель кластеризации  DBSCAN --------------------
def BuildDBSCAN(dataset):
    # масштабируем данные так, чтобы получить нулевое среднее и единичную дисперсию
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = StandardScaler()
    #scaler = MinMaxScaler()
    scaler.fit(dataset)
    X_scaled = scaler.transform(dataset)
    from sklearn.cluster import DBSCAN
    dbscan = DBSCAN(eps=0.005,min_samples=30)
    clusters = dbscan.fit_predict(X_scaled)
    print("Принадлежность к кластерам:\n{}".format(clusters))
    # выводим принадлежность к кластерам
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap=mglearn.cm2, s=60)
    plt.xlabel("Признак 0")
    plt.ylabel("Признак 1")
#---------------------------------------------------------------------

if __name__ == "__main__":
    csv_path = "D:\dat_05.csv"
    dataset = GetDataSetFromFile(csv_path)
    plt.close()
    BuildKMeansModel(dataset)
    plt.close()
    BuildAgglomerativeClustering(dataset)
    plt.close()
    BuildDBSCAN(dataset)
    plt.close()















