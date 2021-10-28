import sys
print("версия Python: {}".format(sys.version))
import pandas as pd
print("версия pandas: {}".format(pd.__version__))
import matplotlib
print("версия matplotlib: {}".format(matplotlib.__version__))
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
import matplotlib.pyplot as plt

# генерируем набор данных
X, y = mglearn.datasets.make_forge()
# строим график для набора данных
plt.interactive(True)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Класс 0", "Класс 1"], loc=4)
plt.xlabel("Первый признак")
plt.ylabel("Второй признак")
print("форма массива X: {}".format(X.shape))
plt.close()


X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.show()
plt.ylim(-3, 3)
plt.xlabel("Признак")
plt.ylabel("Целевая переменная")
plt.close()


#------------------ Классификация. Метод KNeighbors ----------------------------------------

mglearn.plots.plot_knn_classification(n_neighbors=1)
plt.close()
mglearn.plots.plot_knn_classification(n_neighbors=3)
plt.close()

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print("Прогнозы на тестовом наборе: {}".format(clf.predict(X_test)))
print("Правильность на тестовом наборе: {:.2f}".format(clf.score(X_test, y_test)))

# Анализ KNeighborsClassifier
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
# создаем объект-классификатор и подгоняем в одной строке
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("количество соседей:{}".format(n_neighbors))
    ax.set_xlabel("признак 0")
    ax.set_ylabel("признак 1")
axes[0].legend(loc=3)
plt.close()
""" Увеличение числа соседей приводит к сглаживанию границы принятия решений. 
Более гладкая граница соответствует более простой модели.
"""

# оценим качество работы модели на обучающем и тестовом наборах с
# использованием разного количества соседей
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
# пробуем n_neighbors от 1 до 10
neighbors_settings = range(1, 11)
for n_neighbors in neighbors_settings:
# строим модель
    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X_train, y_train)
    # записываем правильность на обучающем наборе
    training_accuracy.append(clf.score(X_train, y_train))
    # записываем правильность на тестовом наборе
    test_accuracy.append(clf.score(X_test, y_test))
plt.plot(neighbors_settings, training_accuracy, label="правильность на обучающем наборе")
plt.plot(neighbors_settings, test_accuracy, label="правильность на тестовом наборе")
plt.ylabel("Правильность")
plt.xlabel("количество соседей")
plt.legend()
plt.close()




#-------------------- Линейные модели -------------------------------------

"""Двумя наиболее распространенными алгоритмами линейной
классификации являются:
- логистическая регрессия (logistic regression), реализованная в классе linear_model.LogisticRegression;
 - линейный метод опорных векторов(linear support vector machines), реализованный в классе
  svm.LinearSVC (support vector classifier – классификатор опорных векторов)"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X, y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize=(10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("Признак 0")
    ax.set_ylabel("Признак 1")
axes[0].legend()
plt.close()

mglearn.plots.plot_linear_svc_regularization()
plt.close()


from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg.score(X_test, y_test)))

"""Значение по умолчанию C=1 обеспечивает неплохое качество модели,
правильность на обучающем и тестовом наборах составляет 95%. Однако
поскольку качество модели на обучающем и тестовом наборах примерно
одинако, вполне вероятно, что мы недообучили модель. """

logreg100 = LogisticRegression(C=100).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg100.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg100.score(X_test, y_test)))

"""использование C=100 привело к более высокой правильности на
обучающей выборке, а также немного увеличилась правильность на
тестовой выборке, что подтверждает наш довод о том, что более сложная
модель должна сработать лучше"""

logreg001 = LogisticRegression(C=0.01).fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(logreg001.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(logreg001.score(X_test, y_test)))


plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim(-5, 5)
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
plt.legend()
plt.close()


"""Поскольку LogisticRegression по умолчанию использует L2
регуляризацию, результат похож на результат, полученный при
использовании модели Ridge (рис. 2.12). Большая степень
регуляризации сильнее сжимает коэффициенты к нулю, хотя
коэффициенты никогда не станут в точности равными нулю.
Изучив график более внимательно, можно увидеть интересный
эффект, произошедший с третьим коэффициентом,
коэффициентом признака «mean perimeter». При C=100 и C=1
коэффициент отрицателен, тогда как при C=0.001 коэффициент
положителен, при этом его оценка больше, чем оценка
коэффициента при C=1. Когда мы интерпретируем данную модель,
коэффициент говорит нам, какой класс связан с этим признаком.
Возможно, что высокое значение признака «texture error» связано
с примером, классифицированным как «злокачественный».
Однако изменение знака коэффициента для признака «mean
perimeter» означает, что в зависимости от рассматриваемой
модели высокое значение «mean perimeter» может указывать либо
на доброкачественную, либо на злокачественную опухоль.
Приведенный пример показывает, что интерпретировать
коэффициенты линейных моделей всегда нужно с осторожностью и скептицизмом."""


for C, marker in zip([0.001, 1, 100], ['o', '^', 'v']):
    lr_l1 = LogisticRegression(C=C, penalty="l1").fit(X_train, y_train)
    print("Правильность на обучении для логрегрессии l1 с C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_train, y_train)))
    print("Правильность на тесте для логрегрессии l1 с C={:.3f}: {:.2f}".format(
    C, lr_l1.score(X_test, y_test)))
plt.plot(lr_l1.coef_.T, marker, label="C={:.3f}".format(C))
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.xlabel("Индекс коэффициента")
plt.ylabel("Оценка коэффициента")
plt.ylim(-5, 5)
plt.legend(loc=3)
plt.close()



# !!!!!! Многие линейные модели классификации предназначены лишь для
# бинарной классификации и не распространяются на случай мультиклассовой классификации
# (за исключением логистической регрессии).

from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=42)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.legend(["Класс 0", "Класс 1", "Класс 2"])
plt.close()

linear_svm = LinearSVC().fit(X, y)
print("Форма коэффициента: ", linear_svm.coef_.shape)
print("Форма константы: ", linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.legend(['Класс 0', 'Класс 1', 'Класс 2', 'Линия класса 0', 'Линия класса 1','Линия класса 2'], loc=(1.01, 0.3))
plt.close()


# Прогнозы областей
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha=.7)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
plt.legend(['Класс 0', 'Класс 1', 'Класс 2', 'Линия класса 0', 'Линия класса 1',
'Линия класса 2'], loc=(1.01, 0.3))
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")
plt.close()

"""Основной параметр линейных моделей – параметр регуляризации,
называемый alpha в моделях регрессии и C в LinearSVC и
LogisticRegression. Большие значения alpha или маленькие значения C
означают простые модели. Конкретно для регрессионных моделей
настройка этих параметров имеет весьма важное значение. Как правило,
поиск C и alpha осуществляется по логарифмической шкале. Кроме того
вы должны решить, какой вид регуляризации нужно использовать: L1
или L2. Если вы полагаете, что на самом деле важны лишь некоторые
признаки, следует использовать L1. В противном случае используйте
установленную по умолчанию L2 регуляризацию. Еще L1 регуляризация
может быть полезна, если интерпретируемость модели имеет важное
значение. Поскольку L1 регуляризация будет использовать лишь
несколько признаков, легче будет объяснить, какие признаки важны для
модели и каковы эффекты этих признаков.
Линейные модели очень быстро обучаются, а также быстро
прогнозируют. Они масштабируются на очень большие наборы данных,
а также хорошо работают с разреженными данными. При работе с
данными, состоящими из сотен тысяч или миллионов примеров, вас,
возможно, заинтересует опция solver='sag' в LogisticRegression и
Ridge, которая позволяет получить результаты быстрее, чем настройки
по умолчанию. Еще пара опций – это класс SGDClassifier и класс
SGDRegressor, реализующие более масштабируемые версии описанных
здесь линейных моделей.
Еще одно преимущество линейных моделей заключается в том, что
они позволяют относительно легко понять, как был получен прогноз, при
помощи формул, которые мы видели ранее для регрессии и
классификации. К сожалению, часто бывает совершенно не понятно,
почему были получены именно такие коэффициенты. Это особенно
актуально, если ваш набор данных содержит высоко коррелированные
признаки, в таких случаях коэффициенты сложно интерпретировать.
Как правило, линейные модели хорошо работают, когда количество
признаков превышает количество наблюдений. Кроме того, они часто
используются на очень больших наборах данных, просто потому, что не
представляется возможным обучить другие модели. Вместе с тем в
низкоразмерном пространстве альтернативные модели могут показать
более высокую обобщающую способность. """





#---------------------- Наивные байесовские классификаторы --------------------------------------
"""В scikit-learn реализованы три вида наивных байесовских классификаторов: 
GaussianNB, BernoulliNB и MultinomialNB. 
GaussianNB можно применить к любым непрерывным данным,
в то время как BernoulliNB принимает бинарные данные,
MultinomialNB принимает счетные или дискретные данные """

X = np.array([[0, 1, 0, 1],
              [0, 0, 0, 1],
              [1, 0, 1, 1],
              [1, 0, 1, 0]])
y = np.array([0, 1, 0, 1])
counts = {}
for label in np.unique(y):
# итерируем по каждому классу подсчитываем (суммируем) элементы 1 по признаку
    counts[label] = X[y == label].sum(axis=0)
print("Частоты признаков:\n{}".format(counts))

"""GaussianNB в основном используется для данных с очень высокой
размерностью, тогда как остальные наивные байесовские модели широко
используются для разреженных дискретных данных, например, для
текста. MultinomialNB обычно работает лучше, чем BernoulliNB, особенно
на наборах данных с относительно большим количеством признаков,
имеющих ненулевые частоты (т.е. на больших документах)."""




#----------------------- Ядерный метод опорных векторов --------------------------------

X, y = make_blobs(centers=4, random_state=8)
y = y % 2
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")

from sklearn.svm import LinearSVC
linear_svm = LinearSVC().fit(X, y)
mglearn.plots.plot_2d_separator(linear_svm, X)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")

"""Линейная модель классификации может отделить точки только с
помощью прямой линии и не может дать хорошее качество для этого
набора данных"""


# добавляем второй признак, возведенный в квадрат
X_new = np.hstack([X, X[:, 1:] ** 2])
from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
# визуализируем в 3D
ax = Axes3D(figure, elev=-152, azim=-26)
# сначала размещаем на графике все точки с y == 0, затем с y == 1
mask = y == 0
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
cmap=mglearn.cm2, s=60)
ax.set_xlabel("признак0")
ax.set_ylabel("признак1")
ax.set_zlabel("признак1 ** 2")

plt.close()

linear_svm_3d = LinearSVC().fit(X_new, y)
coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# показать границу принятия решений линейной модели
figure = plt.figure()
ax = Axes3D(figure, elev=-152, azim=-26)
xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
XX, YY = np.meshgrid(xx, yy)
ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
cmap=mglearn.cm2, s=60)
ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
cmap=mglearn.cm2, s=60)
ax.set_xlabel("признак0")
ax.set_ylabel("признак1")
ax.set_zlabel("признак1 ** 2")


ZZ = YY ** 2
dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
plt.contourf(XX, YY, dec.reshape(XX.shape), levels=[dec.min(), 0, dec.max()],
cmap=mglearn.cm2, alpha=0.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")



"""Из вышесказанного можно сделать вывод, что добавление нелинейных
признаков может улучшить прогнозную силу линейной модели. Однако
часто мы не знаем, какие признаки необходимо добавить, и добавление
большего числа признаков (например, рассмотрение всех возможных
взаимодействий в 100-мерном пространстве признаков) может очень
сильно увеличить стоимость вычислений. К счастью, есть хитрый
математический трюк, который позволяет нам обучить классификатор в
многомерном пространстве, фактически не прибегая к вычислению
нового, возможно, очень высокоразмерного пространства. Этот трюк
известен под названием «ядерный трюк» (kernel trick) и он
непосредственно вычисляет евклидовы расстояния (более точно,
скалярные произведения точек данных), чтобы получить расширенное
пространство признаков без фактического добавления новых признаков."""

from sklearn.svm import SVC
X, y = mglearn.tools.make_handcrafted_dataset()
svm = SVC(kernel='rbf', C=10, gamma=0.1).fit(X, y)
mglearn.plots.plot_2d_separator(svm, X, eps=.5)
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# размещаем на графике опорные векторы
sv = svm.support_vectors_
# метки классов опорных векторов определяются знаком дуальных коэффициентов
sv_labels = svm.dual_coef_.ravel() > 0
mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
plt.xlabel("Признак 0")
plt.ylabel("Признак 1")


# Настройка параметров
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for ax, C in zip(axes, [-1, 0, 3]):
    for a, gamma in zip(ax, range(-1, 2)):
        mglearn.plots.plot_svm(log_C=C, log_gamma=gamma, ax=a)
axes[0, 0].legend(["class 0", "class 1", "sv class 0", "sv class 1"],ncol=4, loc=(.9, 1.2))




X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, random_state=0)
svc = SVC()
svc.fit(X_train, y_train)
print("Правильность на обучающем наборе: {:.2f}".format(svc.score(X_train, y_train)))
print("Правильность на тестовом наборе: {:.2f}".format(svc.score(X_test, y_test)))

plt.plot(X_train.min(axis=0), 'o', label="min")
plt.plot(X_train.max(axis=0), '^', label="max")
plt.legend(loc=4)
plt.xlabel("Индекс признака")
plt.ylabel("Величина признака")
plt.yscale("log")

"""
Один из способов решения этой проблемы – масштабирование всех
признаков таким образом, чтобы все они имели примерно один и тот же
масштаб. Общераспространенный метод масштабирования для ядерного
SVM заключается в масштабировании данных так, чтобы все признаки
принимали значения от 0 до 1. Мы увидим, как это делается с помощью
метода предварительной обработки MinMaxScaler в главе 3, в которой
дадим более подробную информацию. А сейчас давайте сделаем это
«вручную»:
"""
# вычисляем минимальное значение для каждого признака обучающего набора
min_on_training = X_train.min(axis=0)
# вычисляем ширину диапазона для каждого признака (max - min) обучающего набора
range_on_training = (X_train - min_on_training).max(axis=0)
# вычитаем минимальное значение и затем делим на ширину диапазона
# min=0 и max=1 для каждого признака
X_train_scaled = (X_train - min_on_training) / range_on_training
print("Минимальное значение для каждого признака\n{}".format(X_train_scaled.min(axis=0)))
print("Максимальное значение для каждого признака\n {}".format(X_train_scaled.max(axis=0)))

# используем ТО ЖЕ САМОЕ преобразование для тестового набора,
# используя минимум и ширину диапазона из обучающего набора (см. главу 3)
X_test_scaled = (X_test - min_on_training) / range_on_training

svc = SVC()
svc.fit(X_train_scaled, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(
svc.score(X_train_scaled, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(svc.score(X_test_scaled, y_test)))

svc = SVC(C=1000)
svc.fit(X_train_scaled, y_train)
print("Правильность на обучающем наборе: {:.3f}".format(
svc.score(X_train_scaled, y_train)))
print("Правильность на тестовом наборе: {:.3f}".format(svc.score(X_test_scaled, y_test)))







