import numpy as np
import copy


class DecisionTree:
    def __init__(self, max_depth=np.inf):
        self.max_depth = max_depth

    def fit(self, X, t):
        # TODO split X into discrete and cont. columns
        build_tree(X, t, depth=0, parent_node=None, child="yes", gini_index=np.inf)

    def build_tree(X, t, depth, parent_node, child, gini_index):
        # TODO idea: maybe remove option and tell if yes or no based on X?
        if (depth == max_depth) or (gini_index == 0):
            parent_node.add_result(t)
            return
        else:
            depth += 1
            gini_indices, threshold = self._get_gini_index_for_columns(X, t)
            gini_index = np.min(gini_indices)
            node_column = X[:, np.argmin(gini_indices)]
            yes, no, t_yes, t_no = self.split_dataset(X, t, node_column)

            if option == "yes":
                node = self.add_yes(parent_node, column, threshold)
            else:
                node = self.add_no(parent_node, column, threshold)

            build_tree(yes, t_yes, depth, node, "yes", gini_index)
            build_tree(no, t_no, depth, node, "no", gini_index)
        
        # TODO write add_result()
        # TODO write split_dataset()
        # TODO write add_yes()
        # TODO write add_no()



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

                threshold = pairwise_averages[np.argmin(pairwise_gini_indices)]

                # select lowest gini index to represent this column
                gini_index = min(pairwise_gini_indices)
                gini_indices[column] = gini_index
            else:
                threshold = None
                gini_index = self._get_weighted_gini_index(X_column, t)
                gini_indices[column] = gini_index

        return gini_indices, threshold

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
