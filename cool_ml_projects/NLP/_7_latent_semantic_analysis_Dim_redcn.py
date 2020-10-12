# we will be using Truncated Singular Value Decomposition for dimensionality reduction.
# THIS IS AN UNSUPERVISED LEARNING.
from __future__ import print_function, division
import numpy as np
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

"""
we are going to cluster the book titles according to their possible Genre maybe?
we are going to plot the reduced column names from the text
"""
stop_words = stopwords.words('english')
stop_words = stop_words + ['introduction', 'edition', 'series', 'application',
                                'approach', 'card', 'access', 'package', 'plus', 'etext',
                                'brief', 'vol', 'fundamental', 'guide', 'essential', 'printed',
                                'third', 'second', 'fourth']

# tokenize and pre-processing words
lemmatizer = WordNetLemmatizer()
def my_tokenizer(s):
    s = s.lower()
    tokens = RegexpTokenizer(r'\w+').tokenize(s)
    tokens = [token for token in tokens if len(token) > 2]
    document = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    document = [token for token in document if not any(c.isdigit() for c in token)]
    return document


titles = [line.rstrip() for line in open('./datasets/all_book_titles.txt')]

# build list of titles, word_map_index, index_word_map, list of tokens
word_index_map = dict()
index_word_map = list()
current_index = 0
all_tokens = list()
all_titles = list()
error_count = 0

for title in titles:
    try:
        title = title.encode('ascii', 'ignore').decode('utf-8')
        all_titles.append(title)
        tokens = my_tokenizer(title)
        all_tokens.append(tokens)
        for token in tokens:
            if token not in word_index_map:
                word_index_map[token] = current_index
                current_index += 1
                index_word_map.append(token)
    except Exception as e:
        error_count += 1

print("Number of errors parsing file:", error_count, "number of lines in file:", len(titles))
if error_count == len(titles):
    print("There is no data to do anything with! Quitting...")
    exit()


# make document
def tokens_to_vector(tokens):
    x = np.zeros(len(word_index_map))
    for t in tokens:
        index = word_index_map[t]
        x[index] += 1
    return x


token_count = len(word_index_map)
doc_count = len(titles)

# build term_document_matrix NOT document_term_matrix.
X = np.zeros((token_count, doc_count))
i = 0
for tokens in all_tokens: # all tokens has list of documents
    X[:, i] = tokens_to_vector(tokens)
    i += 1


def main():
    svd = TruncatedSVD()
    Z = svd.fit_transform(X)
    print(Z.shape)
    plt.scatter(Z[:, 0], Z[:, 1])
    for i in range(token_count):
        plt.annotate(s=index_word_map[i], xy=(Z[i, 0], Z[i, 1]))
    plt.show()


if __name__ == "__main__":
    main()

