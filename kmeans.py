import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs # make_blobs helps generating datasets 
from sklearn.cluster import KMeans
data = make_blobs(n_samples=200, centers=4, n_features=2, cluster_std=1.6, random_state=50)
points = data[0]
#print(points)
kmeans = KMeans(n_clusters=4) 
kmeans.fit(points) 
plt.scatter(data[0][:, 0], data[0][:, 1]) 
plt.show()
clustering = kmeans.cluster_centers_ 
print(clustering)
clusters = kmeans.fit_predict(points) 
print(clusters)
#coloring clusters
plt.scatter(points[clusters == 0,0], points[clusters == 0,1], s=50, color='red') 
plt.scatter(points[clusters == 1,0], points[clusters == 1,1], s=50, color='blue') 
plt.scatter(points[clusters == 2,0], points[clusters == 2,1], s=50, color='yellow') 
plt.scatter(points[clusters == 3,0], points[clusters == 3,1], s=50, color='green') 
#plotting centers
plt.scatter(clustering[0][0], clustering[0][1], s=50, color='black') 
plt.scatter(clustering[1][0], clustering[1][1], s=50, color='black') 
plt.scatter(clustering[2][0], clustering[2][1], s=50, color='black') 
plt.scatter(clustering[3][0], clustering[3][1], s=50, color='black') 
plt.show()