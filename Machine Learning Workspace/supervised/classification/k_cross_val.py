import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV

'''
We shall be using dimensionality reduction using Kernel PCA.
We shall be using kfold cross validation to get a mean score for our training set.
And we will also we using grid search to decide the optimal parameter values for the model. 
'''
data = pd.read_csv('./datasets/Social_Network_Ads.csv')
data = data.drop(columns=['Gender'])

X = data.iloc[:, :-1]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# as euclidian distance is used we need to scale
x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)




# dim-reduction (kernel pca)
kpca = KernelPCA(kernel='linear', n_components=2)
X_train = kpca.fit_transform(X_train)
X_test = kpca.transform(X_test)

classifier = SVC(kernel='linear', random_state=0)  # poly, logistic, linear, rbf #, gamma=1/X_train.n
classifier.fit(X_train, y_train)




# k-cross-fold-validataion
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)  # cv is dataset divide param.
print('\r\nmean accuracy', accuracies.mean(), '\r\nStd deviation', accuracies.std())

# grid-search-cv
# make a json
parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'kernel': ['linear'], 'gamma': np.linspace(0, 1, 11).tolist()}
]
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10, n_jobs=-1)
grid_search = grid_search.fit(X_train, y_train)
print('\nbest estimator', grid_search.best_estimator_, '\nbest parameter ', grid_search.best_params_, '\nbest score ', grid_search.best_score_)




y_prediction = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_prediction)
print('\r\ny_prediction = ', y_prediction, '\r\nConfusion matrix = ', cm, '\r\n')
# contourf graph
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max() + 1, step=0.5))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('black', 'green')))
for i, j in enumerate(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('cyan', 'magenta'))(i), label=j)

plt.title('KNN classification for happiness of men ')
plt.xlabel('Kpca-1')
plt.ylabel('Kpca-2')
plt.legend()
plt.show()












