import numpy as np


class DecisionTree:
    def __init__(self):
        pass

    def fit(self, X, t):
        # decide which column becomes root
        gini_indices = []
        for X_column in X.shape[1]:
            if self._is_numeric(X_column):
                # sort column
                sorted_X_column = np.sort(X_column)
                # calculate pairwise averages
                pairwise_averages = np.zeros(sorted_X_column.shape - 1, dtype=float)
                for current_value in sorted_X_column.shape[1] - 1:
                    next_value = current_value + 1
                    pairwise_averages[current_value] = (
                        sorted_X_column[current_value] + sorted_X_column[next_value]
                    ) / 2

                # calculate gini index for < pairwise average
                # select lowest gini index to represent this column

            gini_index = self._get_weighted_gini_index(X_column, t)
            gini_indices.append(gini_index)

    def predict(self, X):
        pass

    def _get_weighted_gini_index_numeric(self, X_column, t):
        # write a method to calculate gini index in numeric case
        # TODO generalize with discrete method later :)
        pass

    def _get_weighted_gini_index(self, X_column, t):
        # get weights for weighted gini index
        X_values, X_counts = np.unique(X_column, return_counts=True)
        weights = X_counts / sum(X_counts)
        gini_indices = np.zeros(X_values.shape)

        for i in range(len(X_values)):
            # get probabilities of X value & target value
            X_values_indices = np.where(X_column == X_values[i])[0]
            t_values = t[X_values_indices]
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

    def _is_numeric(self, X_column):
        # assumes only float values are continous numeric values
        if X_column.dtype == "float":
            return true
        else:
            return false
