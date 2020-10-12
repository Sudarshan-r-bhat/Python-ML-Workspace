import numpy as np
from future.utils import iteritems
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup

"""
THIS IS A SUPERVISED LEARNING ALGORITHM

we have a XML doc containing customer reviews for Electronic goods.
our task is to classify whether the review is positive or negative.
consider this to be an unsupervised learning.
"""


def review_tokenizer(lines):
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    tokens = tokenizer.tokenize(lines.lower())
    data = [lemmatizer.lemmatize(words, pos='n') for words in tokens if words not in stopwords.words('english')]
    return data


# CountVectorizer implemented and also appending the label
# it gives frequency of each word in the current document
def tokens_to_vector(tokens, label):
    x = np.zeros(len(word_index_map) + 1)
    for token in tokens:
        index = word_index_map[token]
        x[index] += 1
    # scaling
    x = x / x.sum()
    x[-1] = label
    return x


# extract dataset from XML
pos_path = "./datasets/positive.review"
neg_path = "./datasets/negative.review"
positive_parser = BeautifulSoup(open(pos_path).read(), features="html5lib")
positive_reviews = positive_parser.find_all("review_text")

negative_parser = BeautifulSoup(open(neg_path).read(), features="html5lib")
negative_reviews = negative_parser.find_all("review_text")


original_reviews = list()
word_index_map = dict()
current_index = 0
positive_tokens = list()
negative_tokens = list()

# get all reviews in list, reviews as lists of tokens, word_index_map
# aliter:use gensim.corpora.Dictionary and use Dictionary.doc2bow() to get the word_index_map and BagOfWords. refer: _5_topic_modelling.py
for review in positive_reviews:
    original_reviews.append(review.text)  # as review is in bs4 format
    tokens = review_tokenizer(review.text)
    positive_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map.keys():
            word_index_map[token] = current_index
            current_index += 1

for review in negative_reviews:
    original_reviews.append(review.text)  # as review is in bs4 format
    tokens = review_tokenizer(review.text)
    negative_tokens.append(tokens)
    for token in tokens:
        if token not in word_index_map.keys():
            word_index_map[token] = current_index
            current_index += 1
print("length of word_index_map ", len(word_index_map))

N = len(positive_tokens) + len(negative_tokens)

data = np.zeros((N, len(word_index_map) + 1))
i = 0
for tokens in positive_tokens:
    xy = tokens_to_vector(tokens, 1)
    data[i, :] = xy
    i += 1

for tokens in negative_tokens:
    xy = tokens_to_vector(tokens, 0)
    data[i, :] = xy
    i += 1

original_reviews, data = shuffle(original_reviews, data)

X = data[:, :-1]
y = data[:, -1]


X_train = X[: -100, ]
y_train = y[: -100, ]
X_test = X[-100:, ]
y_test = y[-100:, ]

model = LogisticRegression()
model.fit(X_train, y_train)
print("training accuracy = ", model.score(X_train, y_train))
print("testing accuracy = ", model.score(X_test, y_test))


####################################### below are for better understanding ############################################
threshold = 0.5
for word, index in iteritems(word_index_map):
    weight = model.coef_[0][index]
    if weight > threshold or weight < -threshold:
        print(word, " ", weight)


# misclassified examples
predictions = model.predict(X)
P = model.predict_proba(X)[:, 1]

minP_whenYis1 = 1
maxP_whenYis0 = 0

wrong_positive_review = None
wrong_negative_review = None

wrong_positive_prediction = None
wrong_negative_prediction = None

for i in range(N):
    p = P[i]
    Y = y[i]
    # extract least probable word, review & prediction
    if Y == 1 and p < 0.5:
        if p < minP_whenYis1:
            wrong_positive_review = original_reviews[i]
            wrong_positive_prediction = predictions[i]
            minP_whenYis1 = p
    elif Y == 0 and p > 0.5:
        if p > maxP_whenYis0:
            wrong_negative_review = original_reviews[i]
            wrong_negative_prediction = predictions[i]
            maxP_whenYis0 = p

print("Most wrong positive review (prob = %s, pred = %s):" % (minP_whenYis1, wrong_positive_prediction))
print(wrong_positive_review)
print("Most wrong negative review (prob = %s, pred = %s):" % (maxP_whenYis0, wrong_negative_prediction))
print(wrong_negative_review)


