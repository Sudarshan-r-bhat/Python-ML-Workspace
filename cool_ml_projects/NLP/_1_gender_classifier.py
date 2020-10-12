# import nltk
# nltk.download('names')
import random
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy

"""
The idea here is: we need to know only the last few letter of a name
to predict the gender. ex: kia-ra, kir-an, No-el, No-ah, che-ryl, An-na, jul-ie
"""


def extract_features(word, n=2):
    last_n_letters = word[-n:]
    return {'feature': last_n_letters.lower()}


male_list = [(name, 'male') for name in names.words('male.txt')]
female_list = [(name, 'female') for name in names.words('female.txt')]

dataset = male_list + female_list

# print('the dataset imported is as follows: ')
# for x in dataset:
#     print(x)

random.seed(5)
random.shuffle(dataset)

test_names = ['Alexander', 'Danielle', 'David', 'Cheryl']
num_train = int(0.8 * len(dataset))

for i in range(1, 6):
    print('number of end-letter considerd: ', i)

    # feature extraction
    features = [(extract_features(n, i), gender) for (n, gender) in dataset]

    # train test split
    X_train, X_test = features[: num_train], features[num_train:]

    # classifier and prediction
    classifier = NaiveBayesClassifier.train(X_train)

    accuracy = round(100 * nltk_accuracy(classifier, X_test), 2)
    print("Accuracy : ", accuracy)

    for name in test_names:
        print('names : ', name)
        print('prediction : ', classifier.classify(extract_features(name, i)))
        print('___________________________________________\r\n')

