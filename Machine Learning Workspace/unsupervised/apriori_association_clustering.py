import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apyori import apriori
"""
dataset is a record from a week of transactions.
we are going to build a recommendation system for a French super-market.
based on association rules.
"""
data = pd.read_csv('./datasets/market_basket_association_clust.csv', header=None)

transacts = list()

for i in range(data.shape[0]):
    # transacts should be a list of lists in STRING format
    transacts.append([str(data.values[i, j] for j in range(data.shape[1]))])


# create a sparse matrix, predict rules.
# params: transactions, min support(manually), min confidence, min lift, minimum no of variables you want to compare.
association_rules = apriori(transacts, min_support=0.003, min_confidence=0.2, min_lift=0.3, min_length=2)
result = list(association_rules)
print(result)


