import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

dataset = pd.read_csv("./datasets/Test.csv")

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# print(X.shape)
# encode col: 0, 1, 3
encoder0 = LabelEncoder()
encoder1 = LabelEncoder()
encoder3 = LabelEncoder()

X[:, 0] = encoder0.fit_transform(X[:, 0])
X[:, 1] = encoder1.fit_transform(X[:, 1])
X[:, 3] = encoder3.fit_transform(X[:, 3])

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0, 1, 3])], remainder='passthrough')
X = ct.fit_transform(X)
# print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(X.toarray(), y, test_size=0.2, shuffle=True, random_state=0)

classifier = GaussianNB()  # XGBClassifier()     # GaussianNB() acc: 92%,
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)
cm = confusion_matrix(y_test, prediction)

# print("confusion matrix: \r\n", cm, "\r\nAccuracy:", accuracy_score(y_test, prediction) * 100, "%")

accuracy_score_res = accuracy_score(y_test, prediction) * 100
[l, m], [n, o] = cm
log_result = "confusion matrix: \r\n" + f"[{l}, {m}]\n[{n}, {o}]" + "\r\nAccuracy: " + accuracy_score_res.__str__()

print(log_result)

# print(X_test[:, 34])
################################### Plotting Graphs ################################################


x_set, y_set, z_set = X_test[:, 34], X_test[:, 36], X_test[:, 37]


for i in range(3):
    if i == 0:
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_set[y_test == 1], y_set[y_test == 1], z_set[y_test == 1], c='r', s=2, label="attack found")
        ax.scatter(x_set[y_test == 0], y_set[y_test == 0], z_set[y_test == 0], c='black', s=2, label="Normal")
        ax.set_xlabel('ports')
        ax.set_ylabel('packet length')
        ax.set_zlabel('packets/Time')
        ax.legend()
        fig.set_size_inches(15, 15)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)
        ax.tick_params(axis="z", labelsize=18)
        plt.savefig("./output/fig1.png", orientation='landscape')
    if i == 1:
        fig2 = plt.figure(1)
        bx = fig2.add_subplot(111, projection='3d')
        bx.scatter(y_set[y_test == 1], z_set[y_test == 1], c='cyan', s=1, label="attack found")
        bx.scatter(y_set[y_test == 0], z_set[y_test == 0], c='green', s=1, label="Normal")
        bx.legend()
        bx.set_xlabel("packet length")
        bx.set_ylabel("packet / Time")
        # plt.subplots_adjust(left=0, bottom=0, right=1000, top=1000)  #lbrt
        fig2.set_size_inches(15, 15)
        bx.tick_params(axis="x", labelsize=18)
        bx.tick_params(axis="y", labelsize=18)
        plt.savefig("./output/fig2.png", orientation='landscape')
    if i == 2:
        fig3 = plt.figure(1)
        labels = ['true pos', 'true neg', 'false pos', 'false neg']
        height = cm[0].tolist() + cm[1].tolist()

        # position the subplot
        cx = fig3.add_axes([0.1, 0.1, 0.9, 0.9])  # syntax: left, right, height, width

        cx.bar(labels, height, width=0.5,  color=['blue', 'green', 'yellow', 'cyan'])

        fig3.suptitle("confusion matrix")
        cx.set_xticklabels(labels)
        cx.set_xlabel("labels")
        cx.set_ylabel("frequency")
        fig3.set_size_inches(15, 15)
        cx.tick_params(axis="x", labelsize=22)
        cx.tick_params(axis="y", labelsize=22)
        plt.savefig("./output/fig3.png", orientation='landscape')
        #plt.show()
