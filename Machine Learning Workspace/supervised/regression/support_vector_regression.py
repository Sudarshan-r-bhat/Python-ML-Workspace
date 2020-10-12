import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
'''
here we shall be predicting the acc due to gravity using Support vector regression
the procedure is similar for Decision trees(need not scale), Random forest regression as well.
'''
data = pd.read_csv('./datasets/acc_due_gravity.csv')
X = data.iloc[:, 0].values
y = data.iloc[:, 1].values

X_scaler = StandardScaler()
y_scaler = StandardScaler()
X = X_scaler.fit_transform(X.reshape(-1, 1))
y = y_scaler.fit_transform(y.reshape(-1, 1))

# non-linear regressor hence 'rbf'
# regressor = RandomForestRegressor(n_estimator=300)  # n_estimator implies no of trees to be generated.
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# prediction
prediction = y_scaler.inverse_transform(regressor.predict(X_scaler.transform(np.array([[7000000]]))))
# prediction2 = regressor.predict(np.array([[7000000]]))
print(prediction2)
print(prediction, ' is the prediction for distance = 7000000')

# to get a smoother curve
X_grid = np.arange(min(X), max(X), 0.3)
X_grid = X_grid.reshape(-1, 1)

plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Acceleration due to gravity')
plt.xlabel('distance from surface')
plt.ylabel('acc due to gravity')
plt.show()




