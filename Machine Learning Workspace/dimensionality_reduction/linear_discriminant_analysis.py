import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# read csv
data = pd.read_csv('./datasets/Wine.csv')
# data = data.drop(columns=['index', 'wife_age', 'target'])

# separate the data
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# divide the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale the instances. Not y because 0 <= y <= 1
X_scaler = StandardScaler()
y_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

# y_train = y_scaler.fit_transform(y_train.reshape(-1, 1))
# y_test = y_scaler.transform(y_test.reshape(-1, 1))

# dimensionality reduction
lda = LDA(n_components=2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

# classifier
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Prediction & confusion mat
y_prediction = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_prediction)
print(y_prediction, '\r\n', cm)

# contour graph
# make grid of each column in X
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                    np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.5))

# predict arguments: an array => list() => transpose. further, reshape()
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha=0.75,
             cmap=ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('blue', 'black', 'cyan'))(i), label=j)

plt.title('Logistic regression for wine dataset')
plt.xlabel('LDA-1')
plt.ylabel('LDA-2')
plt.legend()
plt.show()
# black should be in green area, blue in red area. if the prediction is correct.
