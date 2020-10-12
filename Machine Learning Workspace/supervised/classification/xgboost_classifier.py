import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, accuracy_score
from xgboost import XGBClassifier
'''
XGB classifier uses decision trees for classification, it uses gradient boosting, its fast,
also its very accurate and popular.
'''

data = pd.read_csv('./datasets/Churn_Modelling_ANN.csv')
X = data.iloc[:, 3: 13].values
y = data.iloc[:, 13].values

# preprocessing
label_encoder_1 = LabelEncoder()
label_encoder_2 = LabelEncoder()
X[:, 1] = label_encoder_1.fit_transform(X[:, 1])
X[:, 2] = label_encoder_2.fit_transform(X[:, 2])

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]
print(X, X.shape)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = XGBClassifier()
classifier.fit(X_train, y_train)


# predictions/ outputs
y_predictions = classifier.predict(X_test)
y_predictions = (y_predictions > 0.5)  # you cant have a float value for conf..mat..

cm = confusion_matrix(y_test, y_predictions)
print(cm, '\r\naccuracy = ', accuracy_score(y_test, y_predictions))