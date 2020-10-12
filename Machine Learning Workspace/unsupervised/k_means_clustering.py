import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('./datasets/Mall_Customers_unsupervised.csv')

X = data.iloc[:, [3, 4]].values
wcss = list()

for i in range(1, 11):
    # params: no of clusters, how to place the centroids, how many times do we vary the centroid positions, no of times
    # to run the k_means algo, whether to randomize the data each time(0 == no).
    k_means = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
    k_means.fit(X)
    wcss.append(k_means.inertia_)  # sum of squares == inertia

plt.plot(range(1, 11), wcss, 'green')
plt.title('elbow method to determine no of clusters to choose ')
plt.xlabel('no of clusters ')
plt.ylabel('wcss ')
plt.xticks(range(1, 11))
plt.show()

k_means = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)
y_kmeans = k_means.fit_predict(X)
print(y_kmeans)

# plot graph
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='magenta', label='cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='green', label='cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='cyan', label='cluster5')
plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1], s=200, c='yellow', label='centroid')
plt.title('mall statistics ')
plt.ylabel('spending score(1-100)')
plt.xlabel('Annual income (k$)')
plt.legend()
plt.show()


