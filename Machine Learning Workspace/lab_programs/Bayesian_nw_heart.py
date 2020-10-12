# this is termwork 7. it predicts heart diseases for various factors.
# i think it is a Supervised learning: classification problem in ML
# refer for better understanding:
# https://github.com/pgmpy/pgmpy_notebook/blob/master/notebooks/9.%20Learning%20Bayesian%20Networks%20from%20Data.ipynb
# section 2, 3, 4, 6 seems very interesting and important.

import csv
import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator  # probabilistic graphical model.
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination
from pgmpy.estimators import MLE

labels = list(csv.reader(open('dataset7names.csv', 'r')))
heartDisease = pd.read_csv('dataset7heart.csv', names=labels[0])
print(heartDisease)
heartDisease = heartDisease.replace('?', np.NAN)
print(heartDisease.head())
model = BayesianModel([('age', 'trestbps'), ('sex', 'trestbps'), ('trestbps', 'heartdisease'),
                       ('heartdisease', 'thalach')])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
infer = VariableElimination(model)
q = infer.query(variables=['heartdisease'], evidence={'age': 40})
print(q['heartdisease'])
q = infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q['heartdisease'])