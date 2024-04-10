import numpy as np
import copy
from Nodes import Node


class DecisionTree:
    def __init__(self, max_depth=np.inf, classification_tree=True, min_inputs=20):
        self.max_depth = max_depth
        self.root = None
        self.classification_tree = classification_tree
        self.min_inputs = min_inputs

    def fit(self, X, t):
        # fit tree using recursive build_tree() method

        if classification_tree:
            self._build_classification_tree(
                X, t, depth=0, gini_index=np.inf, parent_node=None, child_node_type=None
            )
        else:
            self._build_regression_tree(X, t)
            # TODO write _build_regression_tree()

    def _build_classification_tree(X, t, depth, gini_index, parent_node, child_node_type):
        # check stop conditions
        if (depth == max_depth) or (gini_index == 0):
            # classification case
            # find most occuring class value in target vector
            values, counts = np.unique(t, return_counts=True)
            result = values[np.argmax(counts)]

            # add result to leaf node
            parent_node.add_result(result)

            # stop recursive method for branch
            return
        else:
            # get gini indices for each columns
            # find feature with min gini index
            gini_indices, threshold = self._get_gini_index_for_columns(X, t)
            gini_index = np.min(gini_indices)
            X_column = np.argmin(gini_indices)
            # split dataset into yes/no subsets
            X_yes, X_no, t_yes, t_no = self.split_dataset(X, t, X_column)

            # create child node
            child_node = Node(X_column, threshold)

            # connect to parent node or make root
            if depth == 0:
                self.root = child_node
            else:
                parent_node.add_child_node(child_node, child_node_type)

            # increase depth counter
            depth += 1

            # recursive method call
            self._build_tree(
                X_yes, t_yes, depth, child_node, gini_index, child_node_type="yes_node"
            )
            self._build_tree(
                X_no, t_no, depth, child_node, gini_index, child_node_type="no_node"
            )

        # TODO write split_dataset()

    def predict(self, X):
        # start at root
        current_node = self.root

        # traverse down tree until leaf node
        while current_node.has_children():
            feature = current_node.get_X_column()
            if X[feature] >= threshold:
                current_node = current_node.get_yes_child()
            else:
                current_node = current_node.get_no_child()

        # prediction is result stored at leaf node
        prediction = current_node.get_result()
        return prediction

    def _get_gini_index_for_columns(self, X, t):
        num_features = X.shape[1]
        num_inputs = X.shape[0]

        # initialize gini index array
        gini_indices = np.ones(num_features, dtype=float)

        # for every feature
        for feature in range(num_features):
            X_column = X[:, feature]

            # check if feature is numeric
            if self._is_numeric(X_column):
                # sort X_column
                sorted_X_column = np.sort(X_column)
                # calculate pairwise averages
                pairwise_averages = np.zeros(num_inputs - 1, dtype=float)
                for current_index in range(num_inputs - 1):
                    next_index = current_index + 1
                    pairwise_averages[current_index] = (
                        sorted_X_column[current_index] + sorted_X_column[next_index]
                    ) / 2

                # calculate gini index for X_column < pairwise average
                pairwise_gini_indices = []
                for pairwise_avg in pairwise_averages:
                    discrete_X_column = np.where(X_column < pairwise_avg, 1, 0)
                    pairwise_gini_index = self._get_weighted_gini_index(
                        discrete_X_column, t
                    )
                    pairwise_gini_indices.append(pairwise_gini_index)

                # get threshold
                threshold = pairwise_averages[np.argmin(pairwise_gini_indices)]

                # select lowest gini index to represent this feature
                gini_index = min(pairwise_gini_indices)
                gini_indices[feature] = gini_index

            # if discrete feature
            else:
                # set binary threshold
                threshold = 0.5

                # calculate gini index for feature
                gini_index = self._get_weighted_gini_index(X_column, t)
                gini_indices[feature] = gini_index

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
        # assumes no == 0, yes == 1 in discrete case
        if np.array_equal(np.unique(X_column), np.array([0, 1])):
            return False
        else:
            return True
