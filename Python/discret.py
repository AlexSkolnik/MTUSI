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

#label_id;flow_id;"attr 10";"attr 14";"attr 15";"attr 19";"attr 22";"attr 23"
# Файл не содержит заголовков столбцов, поэтому мы передаем header=None
# и записываем имена столбцов прямо в "names"
data = pd.read_csv("D:/Traffic.csv",
                    header=None, index_col=False,
                    delimiter=";",
                    names=['label_id', 'flow_id', 'attr10', 'attr14', 'attr15','attr19', 'attr22', 'attr23'])
# В целях упрощения мы выберем лишь некоторые столбцы
data = data[['label_id', 'flow_id', 'attr22', 'attr23']]
# IPython.display позволяет вывести красивый вывод, отформатированный в Jupyter notebook
print(data.head())

print(data.label_id.value_counts())
print(data.flow_id.value_counts())
print(data.attr22.value_counts())
print(data.attr23.value_counts())

print("Исходные признаки:\n", list(data.columns), "\n")
data_dummies = pd.get_dummies(data)
print("Признаки после get_dummies:\n", list(data_dummies.columns))
data_dummies.head()