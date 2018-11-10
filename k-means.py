import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import sklearn.cluster as sk
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import folium

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
#NumClust = range(1,21)
#kmeans = [KMeans(n_clusters=i, random_state=0) for i in NumClust] #n_clusters - кол-во кластеров
#fit = [kmean.fit(data) for kmean in kmeans]
#clusters = [kmean.predict(data) for kmean in kmeans]
#scores = [kmean.inertia_ for kmean in kmeans]
#print(scores)
#mpl.plot(NumClust, scores)
#mpl.xlim(0,20)
#mpl.xlabel('Количество кластеров')
#mpl.ylabel('Сумма расстояний')
#mpl.show()
###

## K-means с выводом результата на экран в виде графика
#clusters = KMeans(n_clusters=5, max_iter=10000, precompute_distances=True).fit_predict(data)
#np.savetxt( 'clusters.txt', clusters, delimiter=';')
#fig = mpl.figure()
#ax = fig.add_subplot(111, projection='3d')
clusters = np.loadtxt('clusters.txt', dtype = np.float, delimiter=';')
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

#возвращаем обратно к координатам
power = data[:,3]
data = scaler.inverse_transform(data)
#print([(power[i], data[i,3]) for i in range(1,100,1)])
for i in range(data.shape[0]):
    data[i, 1] = data[i, 1] - 90
    data[i, 2] = (data[i, 2] + 180) % 360
    data[i,2] = data[i,2] - 180

#ax.scatter(data[:,1],data[:,2], data[:,0], color = colors , cmap=mpl.hot())

m = folium.Map(
    location=[45.372, -121.6972],
    zoom_start=1,
    tiles='Stamen Terrain'
)
for i in range(data.shape[0]):
    st = 'Глубина: ' + str(data[i,0]) + ' Сила: ' + str(data[i,3])
    folium.CircleMarker(
        location=[data[i,1], data[i,2]],
        popup= st,
        radius = 2,
        fill = True,
        fill_color = colors[i],
        color = colors[i]
    ).add_to(m)

m.save('index.html')

#mpl.savefig('k-means 5 clusters')
#mpl.show()