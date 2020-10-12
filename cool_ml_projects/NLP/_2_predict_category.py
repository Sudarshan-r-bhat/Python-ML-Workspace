import nltk
from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

# to import datasets
category_map = {
    'talk.politics.misc': 'Politics',
    'rec.autos': "Autos",
    'rec.sport.hockey': "Hockey",
    'sci.electronics': 'Electronics',
    'sci.med': 'Medicine'
}

# dataset
dataset = fetch_20newsgroups(subset='train',
                             categories=category_map.keys(),
                             shuffle=True,
                             random_state=5)

print(type(dataset))

# sparse matrix
cv = CountVectorizer()  # min_df, max_df, max_features
X = cv.fit_transform(dataset.data)
print('the dimension of the training sparse matrix : ', X.shape)

# term-freq and inverse-doc-freq TFIdf
tfidf = TfidfTransformer()
X_train = tfidf.fit_transform(X)


# classifier
classifier = MultinomialNB()
classifier.fit(X_train, dataset.target)


# prepare test test
test_dataset = [
    'You need to be careful with cars when you are driving on slippery roads',
    'A lot of devices can be operated wirelessly',
    'Players need to be careful when they are close to goal posts',
    'Political debates help us understand the perspectives of both sides'
]

cv_test = cv.transform(test_dataset)
X_test = tfidf.transform(cv_test)


# predict
predictions = classifier.predict(X_test)

for sentence, result in zip(test_dataset, predictions):
    print('\nInput: ', sentence)
    print('prediction: ', category_map[dataset.target_names[result]])
    print('_____________________________________________\r\n')


