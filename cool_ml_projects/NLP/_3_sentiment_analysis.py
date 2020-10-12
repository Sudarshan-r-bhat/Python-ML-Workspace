# import nltk
# nltk.download('movie_reviews')
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy

# THIS IS A SUPERVISED ALGORITHM
# My only question in this program is how is that we are passing words directly into the classifier? maybe that's the
# beauty of Naive bayes classifier it takes any type of input. and classify based on the conditional probabilities...


def feature_extraction(words):
    return dict([(word, True) for word in words])


# load all positive and negative files from the folder
positive_file_ids = movie_reviews.fileids('pos')
negative_file_ids = movie_reviews.fileids('neg')

# positive_features = list( tuple( dict(feature_contents), target) )
positive_features = [(feature_extraction(movie_reviews.words(fileids=[id])), 'Positive') for id in positive_file_ids]
negative_features = [(feature_extraction(movie_reviews.words(fileids=[id])), 'Negative') for id in negative_file_ids]

print('sample of the dataset: ')
for x in range(2):
    print(positive_features[x])
print('\n\n\n')
# train test split
num_positives = int(0.8 * len(positive_features))
num_negatives = int(0.8 * len(negative_features))

X_train = positive_features[: num_positives] + negative_features[: num_negatives]
X_test = positive_features[num_positives:] + negative_features[num_negatives:]

# classifier
classifier = NaiveBayesClassifier.train(X_train)
print('the accuracy of the classifier is: ', nltk_accuracy(classifier, X_test))

# prediction
print('below are the 15 most informative words: ')
for i, words in enumerate(classifier.most_informative_features()):
    if i == 15:
        break
    print(f"{i + 1}:", words[0])
print('\n\n')


test_reviews = [
        'The costumes in this movie were great',
        'I think the story was terrible and the characters were very weak',
        'People say that the director of the movie is amazing',
        'This is such an idiotic movie. I will not recommend it to anyone.'
    ]
for review in test_reviews:
    print('review: ', review)
    probabilities = classifier.prob_classify(feature_extraction(review.split()))
    prediction = probabilities.max()
    print('prediction : ', prediction, '\n')


