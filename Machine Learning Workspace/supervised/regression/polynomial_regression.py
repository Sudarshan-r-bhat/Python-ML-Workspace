"""
here are going to predict the acceleration due to gravity at any distance away from earth.
given the distance.
"""
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('./datasets/acc_due_gravity.csv')

X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

print(X.reshape(-1, 1), y.reshape(-1, 1))
# make a linear prediction.
lin_regressor = LinearRegression()
lin_regressor.fit(X.reshape(-1, 1), y.reshape(-1, 1))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='red')
plt.plot(X, lin_regressor.predict(X.reshape(-1, 1)), color='blue')
plt.title('Linear curve for Acceleration due to gravity')
plt.xlabel('distance(kms above surface)')
plt.ylabel('acceleration due to gravity')


# add polynomial features to convert linear regression to polynomial regression.
poly_regressor = PolynomialFeatures(degree=2)
poly_X = poly_regressor.fit_transform(X.reshape(-1, 1))
poly_regressor.fit(poly_X, y)

lin_regressor2 = LinearRegression()
lin_regressor2.fit(poly_X, y.reshape(-1, 1))

# prediction
print('the gravity at 7000km is ',
      lin_regressor2.predict(poly_regressor.fit_transform(np.array(7000000).reshape(-1, 1))))

# to get a smooth curve
X_grid = np.arange(min(X), max(X), 0.5)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.subplot(1, 2, 2)
plt.scatter(X, y, color='black')
plt.plot(X_grid, lin_regressor2.predict(poly_regressor.fit_transform(X_grid.reshape(-1, 1))), color='green')
plt.show()




