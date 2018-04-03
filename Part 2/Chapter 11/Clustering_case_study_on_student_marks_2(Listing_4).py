from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import pandas

colnames=['Student','English','Math','Science']
data=pandas.read_csv('CLustering.csv',names=colnames)
Math=data.Math[1:].tolist()
Science=data.Science[1:].tolist()


x1= np.array(Math)
x2= np.array(Science)

plt.plot()
plt.title('Dataset')
plt.xlabel('Math')
plt.ylabel('Science')
plt.scatter(x1, x2)
plt.show()

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = [‘b’, ‘g’, ‘r’]
markers = [‘o’, ‘v’, ‘s’]

wss = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    kmeanModel.fit(X)
    wss.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, ‘euclidean’), axis=1)) / X.shape[0])


# Plot the elbow
plt.plot(K, wss, ‘bx-‘)
plt.xlim([0, 10])

plt.xlabel(‘Number of clusters’)
plt.ylabel(‘WSS’)
plt.title(‘The Elbow Method showing the optimal k=3’)
plt.show()

#Clustering
kmeanModel = KMeans(n_clusters=3).fit(X)
kmeanModel.fit(X)
y_kmeans = kmeanModel.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeanModel.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.title('3 Clusters identified in the dataset')
plt.xlabel('Math')
plt.ylabel('Science')
plt.show()