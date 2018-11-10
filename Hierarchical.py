from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
from sklearn import mixture
from sklearn.decomposition import PCA
import pandas as pd
from pylab import *
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import folium

data = pd.read_csv('quake.csv', sep=',')
data = data.values
for i in range(data.shape[0]):
    data[i,1] = data[i,1] + 90
    data[i,2] = data[i,2] + 180
    data[i,2] = (data[i,2] + 180) % 360

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

data_dist = pdist(data, 'braycurtis')
data_linkage = linkage(data_dist, method='ward')
flat_clusters = fcluster(data_linkage, 5, 'maxclust', depth = 6)
dendrogram(data_linkage)
#mpl.show()

colors = []
for i in range(len(flat_clusters)): #перечисление используемых цветов (костыль для белых точек на белом фоне)
    if (flat_clusters[i] == 0):
        colors.append('red')
    if (flat_clusters[i] == 1):
        colors.append('brown')
    if (flat_clusters[i] == 2):
        colors.append('green')
    if (flat_clusters[i] == 3):
        colors.append('orange')
    if (flat_clusters[i] == 4):
        colors.append('black')
    if (flat_clusters[i] == 5):
        colors.append('coral')
    if (flat_clusters[i] == 6):
        colors.append('blue')
    if (flat_clusters[i] == 7):
        colors.append('purple')

data = scaler.inverse_transform(data)
#print([(power[i], data[i,3]) for i in range(1,100,1)])
for i in range(data.shape[0]):
    data[i, 1] = data[i, 1] - 90
    data[i, 2] = (data[i, 2] + 180) % 360
    data[i,2] = data[i,2] - 180

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

m.save('hirerch.html')