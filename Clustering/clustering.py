# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    # Children of hierarchical clustering
    children = model.children_
    # Distances between each pair of children
    # Since we don't have this information, we can use a uniform one for plotting
    distance = np.arange(children.shape[0])
    # The number of observations contained in each cluster level
    no_of_observations = np.arange(2, children.shape[0]+2)
    # Create linkage matrix and then plot the dendrogram
    linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)
    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)



df = pd.read_excel('data.xlsx', index_col=0) # Загрузка данных из exel  
normalized_df=(df-df.mean())/df.std()  # НОРМАЛИЗУЕМ ДАННЫЕ

X = normalized_df[['X6','X8','X9','X14','X16']]  # Выбираем нужные столбцы
""" K-MEANS"""
kmeans = KMeans(n_clusters=3) # Метод K-means выбираем кол-во кластеров
predict = kmeans.fit_predict(X) # Запускаем k-means и передаем входную таблицу
df['kmeans'] = predict  # создаем новый столбец с названием k-means и записываем полученные результаты


""" WARD"""
clustering = AgglomerativeClustering(linkage='ward',compute_full_tree=True, n_clusters=3)  # Готовим объект кластеризации
model = clustering.fit(X) # запускаем кластеризацию и передаем таблицу X
labels = model.labels_  # Получение массива классов
df['ward'] = labels  # создаем новый столбец с названием ward и записываем полученные результаты
plt.title('Hierarchical Clustering Dendrogram Ward')
plot_dendrogram(model,leaf_rotation=0, labels=model.labels_)  #  рисуем дендрограмму 
plt.show() # показать график


""" COMPLEATE"""
clustering = AgglomerativeClustering(linkage='complete',compute_full_tree=True, n_clusters=3)
model = clustering.fit(X)  # запускаем кластеризацию и передаем X
labels = model.labels_  # Получение массива классов 
df['complete'] = labels # создаем новый столбец с названием compleate и записываем полученные результаты
plt.title('Hierarchical Clustering Dendrogram Complete')  # подпись к графику
plot_dendrogram(model,leaf_rotation=0, labels=model.labels_)  #  рисуем дендрограмму
plt.show()  # показать график

""" Двухмерные данные"""
X=X[['X6','X8']] # Выбираем нужные столбцы

# KMEANS
kmeans = KMeans(n_clusters=3) # Метод K-means выбираем кол-во кластеров
predict = kmeans.fit_predict(X)  # Получаем номера классов

plt.scatter(X['X6'], X['X8'], c=predict) 
plt.title("K-means")
plt.show()


""" DBSCAN """
db = DBSCAN(eps=0.5, min_samples=4).fit(X)
labels = db.labels_
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

plt.scatter(X['X6'], X['X8'], c=labels)
plt.title("DBSCAN N={}".format(n_clusters_))
plt.show()
