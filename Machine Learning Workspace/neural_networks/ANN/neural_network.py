import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
"""
using this dataset we try to predict on various params if a customer will leave the bank or not.
"""
data = pd.read_csv('./datasets/Churn_Modelling_ANN.csv')
X = data.iloc[:, 3: 13].values
y = data.iloc[:, 13].values

# preprocessing
label_encoder_1 = LabelEncoder()
label_encoder_2 = LabelEncoder()
X[:, 1] = label_encoder_1.fit_transform(X[:, 1])
X[:, 2] = label_encoder_1.fit_transform(X[:, 2])

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]
print(X, X.shape)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# scale the data
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

# create model
model = Sequential()

# build the model
model.add(Dense(units=6, input_dim=11, activation='relu', kernel_initializer='uniform'))
model.add(Dropout(0.4))
model.add(Dense(units=6, activation='relu', kernel_initializer='uniform'))
model.add(Dense(units=1, activation='sigmoid', kernel_initializer='uniform'))

# error correction factors to vary weights
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# training
model.fit(X_train, y_train, epochs=30, batch_size=10)

# predictions/ outputs
y_predictions = model.predict(X_test)
y_predictions = (y_predictions > 0.5)  # you cant have a float value for conf..mat..

cm = confusion_matrix(y_test, y_predictions)
print(cm, '\r\naccuracy = ', accuracy_score(y_test, y_predictions))