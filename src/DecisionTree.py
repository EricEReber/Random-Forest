import numpy as np
import copy
from Nodes import Node, ResultNode
from utils import *


class DecisionTree:
    def __init__(self, max_depth=np.inf, classification_tree=True, min_inputs=20):
        self.max_depth = max_depth
        self.root = None
        self.classification_tree = classification_tree
        self.min_inputs = min_inputs

    def fit(self, X, t):
        # fit tree using recursive build_tree() method

        if self.classification_tree:
            self._build_classification_tree(
                X,
                t,
                branch_depth=0,
                # gini_index=np.inf,
                parent_node=None,
                child_node_type=None,
            )
        else:
            self._build_regression_tree(X, t)
            # TODO write _build_regression_tree()

    def predict(self, X):
        # initialize output vector
        prediction = np.zeros((X.shape[0]))

        # for each test input
        for i in range(X.shape[0]):
            # start at root
            current_node = self.root
            next_node = current_node

            # for keeping track of global feature indices
            features = [j for j in range(X.shape[1])]
            is_numeric_feature = [self._is_numeric(X[:, i]) for i in range(X.shape[1])]

            # traverse down tree until leaf node
            while next_node is not None:
                current_node = next_node
                feature = features[current_node.get_X_feature()]
                if not is_numeric_feature[feature]:
                    features.pop(feature)
                    is_numeric_feature.pop(feature)

                threshold = current_node.get_threshold()
                if X[i, feature] >= threshold:
                    next_node = current_node.get_yes_child()
                else:
                    next_node = current_node.get_no_child()

            # prediction is result stored at leaf node
            prediction[i] = current_node.get_result()

        return prediction

    def accuracy(self, t, pred):
        return np.average(t == pred)

    def _build_classification_tree(
        self, X, t, branch_depth, parent_node, child_node_type
    ):

        gini_indices, thresholds = self._get_gini_index_for_columns(X, t)
        gini_index = np.min(gini_indices)

        X_feature = np.argmin(gini_indices)
        threshold = thresholds[X_feature]

        # create and connect child node
        if t.size:
            values, counts = np.unique(t, return_counts=True)
            result = values[np.argmax(counts)]
            child_node = Node(X_feature, threshold, result)

            # connect to parent node or make root
            if branch_depth == 0:
                self.root = child_node
            else:
                parent_node.add_child_node(child_node, child_node_type)

        # increase branch_depth counter
        branch_depth += 1

        # conditions for stopping recursion
        if (branch_depth == self.max_depth) or (gini_index == 0):
            return

        else:
            # split dataset into yes/no datasets
            X_yes, X_no, t_yes, t_no = self._split_dataset(X, t, X_feature, threshold)

            # recursive method call
            self._build_classification_tree(
                X_yes,
                t_yes,
                branch_depth,
                child_node,
                child_node_type="yes_node",
            )
            self._build_classification_tree(
                X_no,
                t_no,
                branch_depth,
                child_node,
                child_node_type="no_node",
            )

    def _split_dataset(self, X, t, X_feature, threshold):
        # the column we will split the dataset on
        X_column = X[:, X_feature]

        # numeric columns are not deleted as they may be
        # used to create future decisions
        if not self._is_numeric(X_column):
            X = np.delete(X, X_feature, axis=1)

        # get indices of rows where feature > threshold
        yes_indices = np.asarray(X_column >= threshold).nonzero()[0]
        X_yes = X[yes_indices, :]
        t_yes = t[yes_indices]

        # get indices of rows where feature < threshold
        no_indices = np.asarray(X_column < threshold).nonzero()[0]
        X_no = X[no_indices, :]
        t_no = t[no_indices]

        return X_yes, X_no, t_yes, t_no

    def _get_gini_index_for_columns(self, X, t):
        num_features = X.shape[1]
        num_inputs = X.shape[0]

        # initialize gini index array
        gini_indices = np.ones(num_features, dtype=float)
        thresholds = np.ones(num_features, dtype=float)

        # for every feature
        for feature in range(num_features):
            X_column = X[:, feature]

            # if feature is numeric
            if self._is_numeric(X_column):

                # if column only has one entry edge case
                if X_column.size == 1:
                    gini_indices[feature] = 0
                    thresholds[feature] = X_column[0]

                else:
                    # sort X_column
                    sorted_X_column = np.sort(X_column)
                    # calculate pairwise averages
                    pairwise_averages = np.zeros(num_inputs - 1, dtype=float)
                    for current_input_index in range(num_inputs - 1):
                        next_input_index = current_input_index + 1
                        pairwise_averages[current_input_index] = (
                            sorted_X_column[current_input_index]
                            + sorted_X_column[next_input_index]
                        ) / 2

                    # calculate gini index for X_column < pairwise average
                    pairwise_gini_indices = []
                    for pairwise_avg in pairwise_averages:
                        discrete_X_column = np.where(X_column < pairwise_avg, 1, 0)
                        pairwise_gini_index = self._get_weighted_gini_index(
                            discrete_X_column, t
                        )
                        pairwise_gini_indices.append(pairwise_gini_index)
                    # TODO ONLY FOR DEBUG
                    # TODO current theory: when X_column only has 1 value left due to numeric values, the pairwise gini indices cannot be calculated, though the gini obviously in this case has to be 0. Suggested fix, exception for X_column.shape == (1,1), but first investigate if this should ever even happen! If so, fix that :) gl tomorrow eric

                    # get threshold
                    thresholds[feature] = pairwise_averages[
                        np.argmin(pairwise_gini_indices)
                    ]

                    # select lowest gini index to represent this feature
                    gini_index = min(pairwise_gini_indices)
                    gini_indices[feature] = gini_index

            # if discrete feature
            else:
                # set binary threshold
                thresholds[feature] = 0.5

                # calculate gini index for feature
                gini_index = self._get_weighted_gini_index(X_column, t)
                gini_indices[feature] = gini_index

        return gini_indices, thresholds

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
        if (
            np.array_equal(np.unique(X_column), np.array([0, 1]))
            or np.array_equal(np.unique(X_column), np.array([0]))
            or np.array_equal(np.unique(X_column), np.array([1]))
        ):
            return False
        else:
            return True
