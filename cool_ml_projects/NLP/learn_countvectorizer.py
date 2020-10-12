import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import brown


def chunker(input, size):
    chunks = input.split(" ")
    count = 0
    output = []
    chunk_list = []

    for word in chunks:
        chunk_list.append(word)
        if count == size:
            output.append(' '.join(chunk_list))
            count = 0
            chunk_list = []
            continue
        count += 1
    return output


# Read the data from the Brown corpus
input_data = ' '.join(brown.words()[:5400])

# Number of words in each chunk
chunk_size = 800

text_chunks = chunker(input_data, chunk_size)


# Extract the document term matrix(sparse matrix)
count_vectorizer = CountVectorizer()
document_term_matrix = count_vectorizer.fit_transform(text_chunks).toarray()
print(document_term_matrix.shape)

# Extract the vocabulary and display it
vocabulary = np.array(count_vectorizer.get_feature_names())
# print("\nVocabulary:\n", vocabulary)


# Generate names for chunks
chunk_names = []
for i in range(len(text_chunks)):
    chunk_names.append('Chunk-' + str(i+1))

# Print the document term matrix
print("\nDocument term matrix:")
formatted_text = '{:>12}' * (len(chunk_names) + 1)

print('\n', formatted_text.format('Word', *chunk_names), '\n')

for word, item in zip(vocabulary, document_term_matrix.T):
    # 'item' is a 'csr_matrix' data structure
    output = [word] + [str(freq) for freq in item.data]
    print(formatted_text.format(*output))
