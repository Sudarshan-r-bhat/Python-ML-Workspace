from nltk.corpus import stopwords
# nltk.download('stopwords')
import pandas as pd
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv('./datasets/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
print(data.head())
corpus = list()

# clean the text data
for i in range(len(data)):
    reviews = re.sub('[^a-zA-Z]', ' ', data.iloc[i, 0])
    reviews = reviews.split()
    ps = PorterStemmer()
    reviews = [ps.stem(word) for word in reviews if word not in set(stopwords.words('english'))]
    reviews = ' '.join(reviews)
    corpus.append(reviews)

# the above cleaning can be done by passing parameters to countvectorizer as well.
# create sparse matrix.
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
print(X.shape)
y = data.iloc[:, 1].values

# classify using naive bayes classifier.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

classifier = GaussianNB()
# classifier = RandomForestClassifier(n_estimators=30, random_state=0, criterion='entropy')
classifier.fit(X_train, y_train)
y_prediction = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_prediction)
print('confusion matrix', cm)
print('\r\naccuracy score = ', accuracy_score(y_test, y_prediction))


