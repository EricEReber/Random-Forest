import numpy as np

class DecisionTree:

    def __init__(self):
        pass

    def fit(self, X, t):
        pass

    def predict(self, X):
        pass

    def _get_probability(self, column, t):
        values, counts = np.unique(column, return_counts=True)
        weights = counts / sum(counts)
        probabilities = 

    def _get_gini_index(X, t):

