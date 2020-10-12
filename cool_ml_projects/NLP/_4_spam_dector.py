import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from wordcloud import  WordCloud

df = pd.read_csv("./datasets/spam.csv", encoding="ISO-8859-1")
# df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df.columns = ["labels", "data"]
print(df)
# create binary labels
# df["b_labels"] = df["labels"].map({"ham": 0, "spam": 1})








