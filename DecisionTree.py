import numpy as np
import copy


class DecisionTree:
    def __init__(self, max_depth):
        self.max_depth = max_depth

    def fit(self, X, t):
        # decide which column becomes root
        gini_indices = self._get_gini_index_for_columns(X, t)
        root_column = X[:, np.argmin(gini_indices)]
        print(root_column)
        # TODO code a tree
        # TODO implement recursive algorithm

    def predict(self, X):
        pass

    def _get_gini_index_for_columns(self, X, t):
        gini_indices = np.ones(X.shape[1], dtype=float)
        for column in range(X.shape[1]):
            X_column = X[:, column]
            if self._is_numeric(X_column):
                # sort column
                sorted_X_column = np.sort(X_column)
                # calculate pairwise averages
                pairwise_averages = np.zeros(sorted_X_column.shape[0] - 1, dtype=float)
                for current_value in range(sorted_X_column.shape[0] - 1):
                    next_value = current_value + 1
                    pairwise_averages[current_value] = (
                        sorted_X_column[current_value] + sorted_X_column[next_value]
                    ) / 2

                # calculate gini index for < pairwise average
                pairwise_gini_indices = []
                for pairwise_avg in pairwise_averages:
                    discrete_X_column = np.where(X_column < pairwise_avg, 1, 0)
                    pairwise_gini_index = self._get_weighted_gini_index(
                        discrete_X_column, t
                    )
                    pairwise_gini_indices.append(pairwise_gini_index)

                # select lowest gini index to represent this column
                gini_index = min(pairwise_gini_indices)
                gini_indices[column] = gini_index
            else:
                gini_index = self._get_weighted_gini_index(X_column, t)
                gini_indices[column] = gini_index

        return gini_indices

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
                gini_index -= probability**2

            gini_indices[i] = gini_index

        # calculate weighted gini index
        weighted_gini_index = gini_indices @ weights.T

        return weighted_gini_index

    def _is_numeric(self, X_column):
        # assumes only float values are continous numeric values
        if X_column.dtype == "float":
            return True
        else:
            return False
