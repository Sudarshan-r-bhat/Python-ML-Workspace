import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import numpy as np

# this program has a dataset of images of numbers and we are going to predict the numbers using svm.SVC() classifier.
# this is the first ML code that i have executed without seeing.
digit = datasets.load_digits()

# print(len(digit.data))

classifier = svm.SVC(gamma=0.0001, C=100)
x, y = digit.data[: -10], digit.target[: -10]
classifier.fit(x, y)

new_axis = np.array(digit.data[7]).reshape(1, -1)  # new_axis was a necessity for prediction; there was a dimension bug. reshape(1, -1) will add a column, which is unknown to us.

print(digit.data[7], '\n', new_axis)
print('prediction: ', classifier.predict(new_axis))

plt.imshow(digit.images[7], cmap=plt.cm.gray_r, interpolation="nearest")
plt.show()
