from scipy.cluster.hierarchy import *
from scipy.spatial.distance import pdist
from sklearn import mixture
from sklearn.decomposition import PCA
import pandas as pd
from pylab import *
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing

data = pd.read_csv('quake.csv', sep=',')
data = data.values
for i in range(data.shape[0]):
    data[i,1] = data[i,1] + 90
    data[i,2] = data[i,2] + 180
    data[i,2] = (data[i,2] + 180) % 360

scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

data_dist = pdist(data, 'euclidean')
data_linkage = linkage(data_dist, method='ward')
flat_clusters = fcluster(data_linkage, 4, 'maxclust', depth = 6)
dendrogram(data_linkage)

fig = mpl.figure()
ax = fig.add_subplot(111, projection='3d')
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
ax.scatter(data[:,1],data[:,2], data[:,0], color = colors, alpha=1.0 , cmap=mpl.hot())
#mpl.savefig('k-means 5 clusters')
mpl.show()