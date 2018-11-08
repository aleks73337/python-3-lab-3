import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import sklearn.cluster as sk
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

data = pd.read_csv("quake.csv", sep=",", dtype=np.float)
data = data.values

for i in range(data.shape[0]):
    data[i, 1] = data[i, 1] + 90
    data[i, 2] = data[i, 2] + 180
    data[i, 2] = (data[i, 2] + 180) % 360

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

np.savetxt("norm_data.csv", data, delimiter=';', fmt='%.3f')

###plotting figure
#fig = mpl.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data[:,1],data[:,2], data[:,0], data[:,3], cmap=mpl.hot())
#mpl.show()
###

###Метод каменной осыпи
#NumClust = range(1,20,1)
#kmeans = [KMeans(n_clusters=i, random_state=0) for i in NumClust] #n_clusters - кол-во кластеров
#fit = [kmeans[i-1].fit(data) for i in NumClust]
#clusters = [kmeans[i-1].predict(data) for i in NumClust]
#scores = [kmeans[i-1].score(data) for i in NumClust]
#mpl.plot(NumClust, scores)
#mpl.savefig('k-means_cluster_numbers')
###


## K-means с выводом результата на экран в виде графика
clusters = KMeans(n_clusters=4, max_iter=10000, precompute_distances=True).fit_predict(data)
fig = mpl.figure()
ax = fig.add_subplot(111, projection='3d')
colors = []
for i in range(len(clusters)): #перечисление используемых цветов (костыль для белых точек на белом фоне)
    if (clusters[i] == 0):
        colors.append('red')
    if (clusters[i] == 1):
        colors.append('brown')
    if (clusters[i] == 2):
        colors.append('green')
    if (clusters[i] == 3):
        colors.append('orange')
    if (clusters[i] == 4):
        colors.append('black')
    if (clusters[i] == 5):
        colors.append('coral')
    if (clusters[i] == 6):
        colors.append('blue')
    if (clusters[i] == 7):
        colors.append('purple')
    if (clusters[i] == 8):
        colors.append('crimson')
ax.scatter(data[:,1],data[:,2], data[:,0], color = colors , cmap=mpl.hot())
#mpl.savefig('k-means 5 clusters')
mpl.show()