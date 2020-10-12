import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# lets get the data ready for cooking.
msg = pd.read_csv('dataset6.csv', names=['message', 'label'])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

X = msg.message
y = msg.labelnum

print('the message and label are: ')
for a, b in zip(X, y):
    print(a, ',', b)

X_train, X_test, y_train, y_test = train_test_split(X, y)

count_vector = CountVectorizer()
xtrain_dtm = count_vector.fit_transform(X_train)

print('the total features extracted using countVectorizer ', xtrain_dtm.shape[1])
df = pd.DataFrame(np.array(xtrain_dtm.toarray()), columns=count_vector.get_feature_names())
# features are the words whose frequencies add weightage to the document you are referring to.   h
print('features are :\n', df)

# the actual cooking happens happens here.
classifier = MultinomialNB()
classifier.fit(xtrain_dtm, y_train)  # this is where we train the data.

xtest_dtm = count_vector.transform(X_test)  # the data has been fit already during the training so we only transform.
predicted = classifier.predict(xtest_dtm)  # this is where we test our data against the trained model.

print('classification result of the testing samples are : ')
for doc, p in zip(X_test, predicted):
    pred = 'pos' if p == 1 else 'neg'
    print('%s -> %s ' % (doc, pred))

print('\nAccuracy metrics')
print('Accuracy of the classifer is', metrics.accuracy_score(y_test, predicted))
print('Recall :', metrics.recall_score(y_test, predicted), metrics.precision_score(y_test, predicted))
print('Confusion matrix')
print(metrics.confusion_matrix(y_test, predicted))
