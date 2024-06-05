import numpy as np
from Nodes import Node

class DecisionTreeClassifier:
    def __init__(self, max_depth=np.inf):
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, t):
        # build tree recursively
        self._build_tree(
            X,
            t,
            branch_depth=0,
            parent_node=None,
            decision=None,
        )

    def predict(self, X):
        # initialize output vector
        prediction = np.zeros((X.shape[0]))

        # for each test input
        for i in range(X.shape[0]):
            # start at root
            current_node = self.root
            next_node = current_node

            # for keeping track of global X_feature_index indices
            features = [j for j in range(X.shape[1])]
            is_numeric_feature = [self._is_numeric(X[:, i]) for i in range(X.shape[1])]
            X_feature_index = 0

            # traverse down tree until leaf node
            while next_node is not None:
                current_node = next_node
                X_feature_index = features[current_node.get_X_feature_index()]
                if not is_numeric_feature[X_feature_index]:
                    features.pop(X_feature_index)
                    is_numeric_feature.pop(X_feature_index)

                X_feature_threshold = current_node.get_X_feature_threshold()
                if X[i, X_feature_index] >= X_feature_threshold:
                    next_node = current_node.get_yes_child()
                else:
                    next_node = current_node.get_no_child()

            # prediction is result stored at leaf node
            prediction[i] = current_node.get_classification_result(X[i, X_feature_index])

        return prediction

    def accuracy(self, t, pred):
        return np.average(t == pred)

    def _build_tree(self, X, t, branch_depth, parent_node, decision):

        # find best place to split dataset
        best_gini_index, X_feature_index, X_feature_threshold = self._get_best_split(
            X, t
        )

        # create and add child node
        child_node = self._create_child_node(X_feature_index, X_feature_threshold, t)
        self._add_child_node(parent_node, child_node, decision, branch_depth)

        # increase branch_depth counter
        branch_depth += 1

        # conditions for stopping recursion
        if (branch_depth == self.max_depth) or (best_gini_index == 0):
            return

        else:
            # split dataset into yes/no datasets
            X_yes, X_no, t_yes, t_no = self._split_dataset(
                X, t, X_feature_index, X_feature_threshold
            )

            # recursive method call if datasets are not none
            if X_yes.size:
                self._build_tree(
                    X_yes,
                    t_yes,
                    branch_depth,
                    child_node,
                    decision=True,
                )
            if X_no.size:
                self._build_tree(
                    X_no,
                    t_no,
                    branch_depth,
                    child_node,
                    decision=False,
                )

    def _create_child_node(self, X_feature_index, X_feature_threshold, t):
        # create and connect child node
        values, counts = np.unique(t, return_counts=True)
        result = values[np.argmax(counts)]
        child_node = Node(X_feature_index, X_feature_threshold, result)

        return child_node

    def _add_child_node(self, parent_node, child_node, decision, branch_depth):
        # connect to parent node or make root
        if branch_depth == 0:
            self.root = child_node
        else:
            parent_node.add_child_node(child_node, decision)

    def _split_dataset(self, X, t, X_feature_index, X_feature_threshold):
        # the column we will split the dataset on
        X_feature = X[:, X_feature_index]

        # numeric columns are not deleted as they may be
        # used to create future decisions
        if not self._is_numeric(X_feature):
            X = np.delete(X, X_feature_index, axis=1)

        # get indices of rows where X_feature_index > X_feature_threshold
        yes_input_indices = np.asarray(X_feature >= X_feature_threshold).nonzero()[0]
        X_yes = X[yes_input_indices, :]
        t_yes = t[yes_input_indices]

        # get indices of rows where X_feature_index < X_feature_threshold
        no_input_indices = np.asarray(X_feature < X_feature_threshold).nonzero()[0]
        X_no = X[no_input_indices, :]
        t_no = t[no_input_indices]

        return X_yes, X_no, t_yes, t_no

    def _get_best_split(self, X, t):
        num_features = X.shape[1]
        num_inputs = X.shape[0]

        # initialize gini index array
        gini_indices = np.ones(num_features, dtype=float)
        thresholds = np.ones(num_features, dtype=float)

        # for every X_feature_index
        for X_feature_index in range(num_features):
            X_feature = X[:, X_feature_index]

            # if X_feature_index is numeric
            if self._is_numeric(X_feature):

                # if column only has one entry edge case
                if X_feature.size == 1:
                    gini_indices[X_feature_index] = 0
                    thresholds[X_feature_index] = X_feature[0]

                else:
                    # sort X_feature
                    sorted_X_column = np.sort(X_feature)
                    # calculate pairwise averages
                    pairwise_averages = np.zeros(num_inputs - 1, dtype=float)
                    for current_input_index in range(num_inputs - 1):
                        next_input_index = current_input_index + 1
                        pairwise_averages[current_input_index] = (
                            sorted_X_column[current_input_index]
                            + sorted_X_column[next_input_index]
                        ) / 2

                    # calculate gini index for X_feature < pairwise average
                    pairwise_gini_indices = []
                    for pairwise_avg in pairwise_averages:
                        discrete_X_column = np.where(X_feature < pairwise_avg, 1, 0)
                        pairwise_gini_index = self._get_weighted_gini_index(
                            discrete_X_column, t
                        )
                        pairwise_gini_indices.append(pairwise_gini_index)

                    # get X_feature_threshold
                    thresholds[X_feature_index] = pairwise_averages[
                        np.argmin(pairwise_gini_indices)
                    ]

                    # select lowest gini index to represent this X_feature_index
                    gini_index = min(pairwise_gini_indices)
                    gini_indices[X_feature_index] = gini_index

            # if discrete X_feature_index
            else:
                # set binary X_feature_threshold
                thresholds[X_feature_index] = 0.5

                # calculate gini index for X_feature_index
                gini_index = self._get_weighted_gini_index(X_feature, t)
                gini_indices[X_feature_index] = gini_index

        best_gini_index = np.min(gini_indices)
        X_feature_index = np.argmin(gini_indices)
        X_feature_threshold = thresholds[X_feature_index]

        return best_gini_index, X_feature_index, X_feature_threshold

    def _get_weighted_gini_index(self, X_feature, t):
        # get weights for weighted gini index
        X_values, X_value_counts = np.unique(X_feature, return_counts=True)
        weights = X_value_counts / sum(X_value_counts)
        gini_indices = np.zeros(X_values.shape)

        for i in range(len(X_values)):
            # get probabilities of X value & target value
            X_values_indices = np.where(X_feature == X_values[i])[0]
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

    def _is_numeric(self, X_feature):
        # assumes no == 0, yes == 1 in discrete case
        if (
            np.array_equal(np.unique(X_feature), np.array([0, 1]))
            or np.array_equal(np.unique(X_feature), np.array([0]))
            or np.array_equal(np.unique(X_feature), np.array([1]))
        ):
            return False
        else:
            return True
