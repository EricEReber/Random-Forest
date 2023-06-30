import numpy as np

class DecisionTree:

    def __init__(self):
        pass

    def fit(self, X, t):
        pass

    def predict(self, X):
        pass

    def _get_weighted_gini_index(self, X_column, t):
        # get weights for weighted gini index
        X_values, X_counts = np.unique(X_column, return_counts=True)
        weights = X_counts / sum(X_counts)
        gini_indices = np.zeros(X_values.shape)

        for i in range(len(X_values)):
            # get probabilities of X value & target value
            X_val_indices = np.where(X_column == X_values[i])[0]
            t_values = t[X_val_indices]
            _, t_counts = np.unique(t_values, return_counts=True)
            probabilities = t_counts / sum(t_counts)

            # get gini index
            gini_index = 1
            for probability in probabilities:
                gini_index -= probability² 

            gini_indices[i] = gini_index

        # calculate weighted gini index
        weighted_gini_index = gini_indices @ weights.T
        
        return weighted_gini_index



