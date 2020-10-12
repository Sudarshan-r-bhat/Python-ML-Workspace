"""
first i will have to get a dendrogram.
then use Agglomertive clustering to cluster the data.
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

data = pd.read_csv('./datasets/Mall_Customers_unsupervised.csv')

# 2 cols as a numpy array values.
X = data.iloc[:, [3, 4]].values

# make a dendrogram. it implements matplotlib internally.
# use this to determine how many clusters to make.
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.xlabel('clustering')
plt.ylabel('euclidean distance')
plt.title('dendrogram_for_mall customers')
plt.show()

# agglomerative clustering aliter: divisive clustering
# ward is like wcss, used to reduce variance in distance calc.
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')

# prediction
y_hc = hc.fit_predict(X)

print('the clustering are as follows\r\n', y_hc)

# plotting the cluster in 2D
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c='red', label='cluster1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c='green', label='cluster2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c='blue', label='cluster3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c='cyan', label='cluster4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c='magenta', label='cluster5')
plt.title('Heirarchical clustering using Agglomerative clust approach for "Mall customers"')
plt.xlabel('Annual Income (k$)')
plt.ylabel('spending score(1-100)')
plt.legend()
plt.show()
