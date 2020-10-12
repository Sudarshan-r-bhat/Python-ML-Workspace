import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# we are predicting the tip for a given bill


def kernel(point, xmat, k):
    m, n = np.shape(xmat)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff = point - X[j] # point and diff are matrices.
        weights[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))
    return weights


def localWeight(point, xmat, ymat, k):
    wt = kernel(point, xmat, k)
    W = (X.T * (wt * X)).I * (X.T * (wt*ymat.T))
    return W


def localWeightRegression(xmat, ymat, k):
    m, n = np.shape(xmat) # xmat defines the x-axis.
    predict_y = np.zeros(m)
    for i in range(m):
        predict_y[i] = xmat[i] * localWeight(xmat[i], xmat, ymat, k)
    return predict_y


data = pd.read_csv('dataset10.csv')
bill = np.array(data.total_bill)
tip = np.array(data.tip)
mbill = np.mat(bill)
mtip = np.mat(tip)
m = np.shape(mbill)[1]
one = np.mat(np.ones(m))

X = np.hstack((one.T, mbill.T)) # it is similar to zip function maps 1 => elements of matrix to get 2-D matrix.
predict_y = localWeightRegression(X, mtip, 3)
graphPlot(X, predict_y)
print('')
