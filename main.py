import numpy as np
import pandas as pd
import matplotlib.pyplot as mpl
import sklearn.cluster as sk
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
data = pd.read_csv("quake.csv", sep=",", dtype=np.float)
data = data.values
for i in range(data.shape[0]):
    data[i,2] = data[i,2] + 150
    data[i,2] = (data[i,2] + 150) % 300 - 150

###plotting figure
#fig = mpl.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(data[:,1],data[:,2], data[:,0], data[:,3], cmap=mpl.hot())
#mpl.show()
###

clusters = KMeans(n_clusters=5, random_state=0).fit_predict(data)
fig = mpl.figure()
ax = fig.add_subplot(111, projection='3d')
colors = []
for i in range(len(clusters)):
    if (clusters[i] == 0):
        colors.append('r')
    if (clusters[i] == 1):
        colors.append('b')
    if (clusters[i] == 2):
        colors.append('g')
    if (clusters[i] == 3):
        colors.append('orange')
    if (clusters[i] == 4):
        colors.append('black')
ax.scatter(data[:,1],data[:,2], data[:,0], color = colors , cmap=mpl.hot())
mpl.show()