'''
we shall predict the happiness of men on a scale of 1- 10
based on the features: married, age, alcoholic, salary, expenditure on family.
'''
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

data_path = "./datasets/happiness_of_men.csv"
data = pd.read_csv(data_path)
data = data.drop(columns=['index', 'wife_age'])

# separating dataset as inputs and outputs.
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
print(X, y)

# split dummy column(similar to multivalued dependency in DBMS)
# onehotencoder will split into cols, label encoder will assign values 0, 1, 2..
# the below would required if the data was {marital status: married/ not married}.
"""
label_encoder_X = LabelEncoder()
X[:, 1] = label_encoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categories=[1])
X = onehotencoder.fit_transform(X).toarray()
print(X)
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
print(np.floor(np.array(y_pred)[:, np.newaxis]))




