"""
article spinner is a technique used to find contexts having similar meaning.
ex: finding duplicate contents from the web-pages by search engines

Here, we will be using all the amazon reviews to search for duplicates, having different words
in the context but similar meaning

actually we are not detecting duplicate instead we are generating duplicate sentences for the original sentences ?
"""
from future.utils import iteritems
import nltk
import random
import numpy as np
from bs4 import BeautifulSoup


parser = BeautifulSoup(open('./datasets/positive.review').read())
positive_reviews = parser.findAll('review_text')

# make trigrams
trigrams = dict()
for review in positive_reviews:
    s = review.text.lower()
    tokens = nltk.tokenize.word_tokenize(s)
    for i in range(len(tokens) - 2):
        k = (tokens[i] + tokens[i + 2])
        if k not in trigrams:
            trigrams[k] = []
        trigrams[k].append(tokens[i + 1])


# for each trigram value, get its word frequency. similar to DOCUMENT_TERM_MATRIX but not the same.
trigram_probabilities = {}  # dict( key: dict(words[:] : frequency[:] ))
for k, words in iteritems(trigrams):
    if len(set(words)) > 1:
        d = {}
        n = 0
        for w in words:
            if w not in d:
                d[w] = 0
            d[w] += 1
            n += 1
        # get probability or simply Normalize
        for w, c in iteritems(d):
            d[w] = float(c) / n
        trigram_probabilities[k] = d


# randomly pick word
def random_sample(d):
    r = random.random()
    cumulative = 0
    for w, p in iteritems(d):
        cumulative += p
        if r < cumulative:
            return w


def test_spinner():
    review = random.choice(positive_reviews)
    s = review.text.lower()
    print("original: ", s)
    tokens = nltk.tokenize.word_tokenize(s)

    for i in range(len(tokens) - 2):
        if random.random() < 0.2:
            k = (tokens[i], tokens[i + 2])
            if k in trigram_probabilities:
                w = random_sample(trigram_probabilities[k])
                tokens[i + 1] = w
    print('spun : ')
    print(" ".join(tokens).replace(" .", ".").replace(" '", "'").replace("$ ", "$").replace(" !", "!"))


if __name__ == '__main__':
    test_spinner()
