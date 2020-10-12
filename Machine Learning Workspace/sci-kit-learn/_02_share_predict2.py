import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm, preprocessing
from matplotlib import style
style.use('ggplot')


def randomizing():
    df = pd.DataFrame({'D1': range(5), 'D2': range(5)})
    df2 = df.reindex(np.random.permutation(list(df.index))) # well we have to convert the df.index(@deprecated) to list.
    print(df2)


def build_dataset_(features=["DE Ratio",
                             "Trailing P/E"]):
    data_df = pd.read_csv("D:\\ML_datasets\\ML_share_dataset\\intraQuarter\\intraQuarter\\key_stats.csv")
    data_df = data_df[: 100]
    X = np.array(data_df[features].values)
    y = data_df['Status'].replace('underperform', 0).replace('outperform', 1).values.tolist()

    X = preprocessing.scale(X)  # preprocessing uses std deviation and other statistical techniques to properly organize the data.
    return X, y


def analysis():

    text_size = 100
    X, y = build_dataset_()
    print(X)
    clf = svm.SVC(kernel='linear', C=1.0)
    clf.fit(X[:, :], y[:])

    correct_count = 0
    for x in range(1, text_size + 1):
        if clf.predict([X[-x]])[0] == y[-x]:  # pass list of list to predict.
            correct_count += 1
    print('Accuracy ', (correct_count / text_size) * 100)


    """
    as we cant plot all the features that we are going to consider. Hence commented out.
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 1]))
    yy = a * xx - clf.intercept_[0] / w[1]
    h0 = plt.plot(xx, yy, "k-", label="non weighted")
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.ylabel("Trailing P/E")
    plt.xlabel("DE Ratio")
    plt.legend()
    plt.show()"""


# analysis()
randomizing()











"""
X = np.array([
    [1, 2],
    [5, 8],
    [1.5, 1.8],
    [8, 8],
    [1, 0.6],
    [9, 11]
])

y = [0, 1, 0, 1, 0, 1]

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

arr = np.array([10.34, 10.34]).reshape(1, -1)
print(clf.predict(arr))

w = clf.coef_
print(w)

a = -w[0][0] / w[0][1]
xx = np.linspace(0, 12) # used to add 10 more numbers at equal interval for fitting the data.
yy = a * xx - clf.intercept_[0] / w[0][1]

h0 = plt.plot(xx, yy, 'k-', label='non weighted divide') # k- means dark lines
print(X[: 0], X[: 1])
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.legend()
plt.show()"""




