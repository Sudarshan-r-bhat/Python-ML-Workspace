import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
'''
this template is similar for all other classification algorithms.
classifier syntax is put in comments below.
'''
data = pd.read_csv('./datasets/happiness_of_men.csv')
data = data.drop(columns=['index', 'wife_age', 'expenditure', 'married', 'alcholic', 'target'])

X = data.iloc[:, :2]
y = data.iloc[:, -1]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

x_scaler = StandardScaler()
X_train = x_scaler.fit_transform(X_train)
X_test = x_scaler.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
# classifier = SVC(kernel='linear', random_state=0, gamma=1/X_train.n) # kernel aliter: poly, rbf(radial basis function)
# classifier = GaussianNB()
# i don't have to scale for Decision trees and Random Forests. coz we are not using Euclidian distance.
# classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
# classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)

y_prediction = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_prediction)
print('y_prediction = ', y_prediction, '\r\nConfusion matrix = ', cm, '\r\n')

# contourf graph
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max() + 1, step=0.5),
                     np.arange(start=X_set[:, 1].min(), stop=X_set[:, 1].max() + 1, step=0.5))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha=0.75, cmap=ListedColormap(('red', 'green')))
for i, j in enumerate(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=ListedColormap(('blue', 'black'))(i), label=j)

plt.title('KNN classification for happiness of men ')
plt.xlabel('age')
plt.ylabel('salary')
plt.legend()
plt.show()

