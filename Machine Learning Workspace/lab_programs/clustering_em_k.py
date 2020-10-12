import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import time

iris=datasets.load_iris()

X=pd.DataFrame(iris.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
y= pd.DataFrame(iris.target)  # we are storing the column name.(?)
y.columns=['Targets']  # and naming the column as Targets.

model=KMeans(n_clusters=3)
model.fit(X)

colormap =np.array(['red','lime','black'])
colormap0 =np.array(['blue', 'orange', 'green'])
plt.subplot(2, 2, 1)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap0[y.Targets], s=10)
plt.title('Real Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

plt.subplot(2, 2, 2)
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[model.labels_], s=10)
plt.title('K-Means Clustering')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
# preprocessing library is used to transform raw data to 'analyzable' form for data processing.
scaler = preprocessing.StandardScaler()
# standardScaler prepares itself to convert the data to plottable points in the range of 0 - 1.
print(scaler)
scaler.fit(X)  # we store the raw data to the scaler. and then transform it to the scaler form.
xsa = scaler.transform(X)
print(xsa)
xs = pd.DataFrame(xsa, columns=X.columns)  # data is converted to list and displayed in tabular format.
print(xs)
gmm = GaussianMixture(n_components=3)
gmm.fit(xs)
gmm_y = gmm.predict(xs)
print(gmm_y)
plt.subplot(223) # 223 is equivalent to 2, 2, 3 => no of rows, columns, index
plt.scatter(X.Petal_Length, X.Petal_Width, c=colormap[gmm_y], s=10)
plt.title('Gmm clustering :')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.show()
time.sleep(5)
print('observation: The GMM using EM algorithm based clustering matched the true labels more closely than the Kmeans')
