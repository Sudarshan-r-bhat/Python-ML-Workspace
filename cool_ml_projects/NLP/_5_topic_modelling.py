from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from gensim.models import LdaModel
from gensim import corpora

# this is an UNSUPERVISED ALGORITHM.
# LDA model: Latent Dirichlet Allocation.


def load_data(path):
    data = list()
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(line[: -1])
    return data


def process(input_text):
    # tokenizer, stopwords, stemmer declaration
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')

    # tokenize
    tokens_ = tokenizer.tokenize(input_text.lower())
    tokens_ = [stemmer.stem(word) for word in tokens_ if word not in stop_words]

    return tokens_


# load and get tokens
path = 'C:\\workstation\\PycharmProjects\\cool_ml_projects\\NLP\\datasets\\topic_modelling.txt'
data = load_data(path)
tokens = [process(line) for line in data]

# make dictonary of tokens {index: token_name},, It creates a word_index_map
dictonary_token = corpora.Dictionary(tokens)
print(dictonary_token.keys(), dictonary_token)

# if the token exits in the dictionary then, returns (document_id: 1), here 1 affirms the presence of a particular token. doc2bow => doc-2-bag-of-words
doc_term_matrix = [dictonary_token.doc2bow(token) for token in tokens]
print(doc_term_matrix)

num_topics = 2

ldamodel = LdaModel(doc_term_matrix, num_topics=num_topics, id2word=dictonary_token, passes=25)

print('\nTop 5 words contributing to each topic : ')
for item in ldamodel.print_topics(num_topics=num_topics, num_words=5):
    print('\nTopic ', item[0])
    list_of_strings = item[1].split("+")  # "1 wt * word + 2...+ 3...+ 4....+ 5_items..."

    for text in list_of_strings:
        weight = text.split("*")[0]
        word = text.split("*")[1]
        print(word, "==>", str(round(float(weight) * 100, 2)), "%")

