# this is a classification algorithm.

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets

iris = datasets.load_iris()
print('Iris dataSets loaded') # we are going to store sepal's and petal's (length and width)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.1) # iris.target will have stdized data.
print('the size of the training data and the size of label = ', x_train.shape, y_train.shape)
print('the size of the testing data and the size of label = ', x_test.shape, y_test.shape)

for i in range(len(iris.target_names)):
    print('labels', i, '=', str(iris.target_names[i]))
classifier = KNeighborsClassifier(n_neighbors=1) # Search for the K observations in the training data that are "nearest" to the measurements of the unknown iris
classifier.fit(x_train, y_train)  # we are training here.
predict_y = classifier.predict(x_test)

print('the results of KNN classification with k = 1 :')
for r in range(len(x_test)):
    print('sample: ', x_test[r], ' original result : ', y_test[r], ' the predicted result : ', predict_y[r])
print('classification accuracy: ', classifier.score(x_test, y_test))


from sklearn.metrics import classification_report, confusion_matrix
print('Confusion Matrix')
print(confusion_matrix(y_test,predict_y))
print('Accuracy Metrics')
print(classification_report(y_test,predict_y))