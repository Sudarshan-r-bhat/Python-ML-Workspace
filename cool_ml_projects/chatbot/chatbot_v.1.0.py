import pandas
import numpy as np
import pandas as pd
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from keras.models import Sequential
from keras.layers import Dense, Dropout  # dropout is a function for the above layer. its not a layer itself.
import matplotlib.pyplot as plt


def train_test_split(dataset, train_percent):
    length = len(dataset)
    decimal = train_percent / 100
    split_marker = math.floor(length * decimal)
    return dataset.text[: split_marker], dataset.text[split_marker:], dataset.target[: split_marker], dataset.target[split_marker: ]


def plot_graph(history0):
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1) # rows, no of graphs in the plot, column index
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)

    plt.plot(history0.history['loss'], label='Training Loss')
    plt.plot(history0.history['val_loss'], label='Testing Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)

    plt.plot(history0.history['acc'], label='Training Accuracy')
    plt.plot(history0.history['val_acc'], label='Testing Accuracy')
    plt.legend()
    plt.show()


dataset = pd.read_csv("dataset_chatbox_v.1.0/bbc-text.csv")  # its a dataframe object.
# print(dataset.head())
print(set(dataset['category']))  # to get all the different classes.
# one-hot classification for multiple classes.  from sklearn.preprocessing import LabelEncoder  can be used.
dataset_new = dataset.replace({'category': {'business': '0', 'tech': '1', 'politics': '2',
                                            'entertainment': '3', 'sport': '4'}})
# print(dataset_new.head())
dataset['target'] = dataset_new['category']
# print(dataset.head())
x, y = dataset['text'], dataset['target']
# print(x, y)

count_vect = CountVectorizer()
x_vect = count_vect.fit_transform(x)
# print(str(x_vect.toarray()))

cd_matrix = np.array(x_vect.toarray())
rows = [f'document_{x}' for x in range(1, cd_matrix.shape[0] + 1)]
cols = count_vect.get_feature_names()
#print(len(cols))
df = pd.DataFrame(data=cd_matrix, index=rows, columns=cols)
transformer = TfidfTransformer()
x_tfidf = transformer.fit_transform(x_vect)

# print(x_tfidf.toarray())
# print(df.loc['document_1']['zuluaga'])
# count_vectorizer = CountVectorizer.fit_transform()

model = Sequential()
model.add(Dense(32, input_dim=x_tfidf.shape[1], activation='relu')) # inp_dim is needed only for input layer.
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))  # we go on adding layers.
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))  # if the goal is classification we either use sigmoid or softmax more often.

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# why did we use loss='parse_categorical_crossentropy' its a best fit for our model?

history = model.fit(x_tfidf, y, 16, epochs=10, validation_split=0.3)
# parameter verbose=0 will make sure progressbar is not shown during the training.
# 16 is batch size.
data = ['the fortnite we played last night was as amazing as always because the fun we had last was to the next level'
        'as we know its one of the most popular computer games.']


p = count_vect.transform(data)
t = transformer.transform(p)
prediction = model.predict(p)

classes = ['business', 'tech', 'politics', 'entertainment', 'sport']
name = classes[np.argmax(prediction)]
print(name)
plot_graph(history)
