import numpy as np
from Nodes import Node


class DecisionTreeRegressor:
    def __init__(self, max_depth=np.inf, min_inputs=7, alpha=0):
        self.max_depth = max_depth
        self.min_inputs = min_inputs
        self.alpha = alpha
        self.root = None
        self.total_squared_error = 0
        self.leaf_nodes = {}
        self.tree_score = 0

    def fit(self, X, t):
        # build tree recursively
        self._build_tree(
            X,
            t,
            branch_depth=0,
            parent_node=None,
            decision=None,
        )
        self.calculate_tree_score()
        # TODO figure out how to do pruning efficiently, because I'd assume I'll need some trackers etc.
        # One option for pruning would be tracking the depth, then calling _build_tree() with decreasing max_depth
        # however, I'm not even sure if this is necessary because it seems pruning happens outside the class
        # since it returns multiple instances of DecisionTreeRegressor, with different numbers of tree nodes.
        # perhaps fit() could return a depth, which could then be used to prune from the outside
        # prune() would then be a utils() function that could be combines with CV() that I suppose would
        # also be in utils.

    def calculate_tree_score(self):
        self.tree_score = self.total_squared_error + self.alpha * len(self.leaf_nodes)

    def get_tree_depth(self):
        # get highest tree depth
        tree_depth = 0
        for leaf_node in self.leaf_nodes:
            if leaf_node.get_node_depth() > tree_depth:
                tree_depth = leaf_node.get_node_depth()

        return tree_depth

    def prune(self):
        tree_depth = get_tree_depth()

        # edge case where we'd prune root since root has no parent
        if tree_depth == 0:
            # TODO maybe raise error
            return tree_depth

        # remove leaf_nodes with highest depth
        for leaf_node in self.leaf_nodes:
            if leaf_node.get_node_depth() == tree_depth:
                # remove parent's children
                parent_node = leaf_node.get_parent_node()
                parent_node.remove_children()

                # add parents to leaf_nodes
                self.leaf_nodes.add(parent_node)

                # remove leaf node from leaf_nodes
                self.leaf_nodes.remove(child_node)

                # subtract squared_error
                self.total_squared_error -= leaf_node.get_squared_error()


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
            prediction[i] = current_node.get_regression_result()

        return prediction

    def _build_tree(self, X, t, branch_depth, parent_node, decision):
        # find best place to split dataset
        X_feature_index, X_feature_threshold, squared_error = self._get_best_split(X, t)

        # create and add child node
        child_node = self._create_child_node(
            X_feature_index, X_feature_threshold, branch_depth, squared_error, t
        )
        self._add_child_node(parent_node, child_node, decision, branch_depth)

        # increase branch_depth counter
        branch_depth += 1

        # split dataset into yes/no datasets
        X_yes, X_no, t_yes, t_no = self._split_dataset(
            X, t, X_feature_index, X_feature_threshold
        )

        # conditions for stopping recursion
        # if max depth reached, if X is too small to split, if no change in X
        if (
            (branch_depth == self.max_depth)
            or (X.shape[0] < self.min_inputs)
            or (np.array_equal(X, X_yes))
            or (np.array_equal(X, X_no))
        ):
            self.leaf_nodes.add(child_node)
            return

        else:
            self.total_squared_error += squared_error
            # recursive method call if datasets are not none
            self._build_tree(
                X_yes,
                t_yes,
                branch_depth,
                child_node,
                decision=True,
            )
            self._build_tree(
                X_no,
                t_no,
                branch_depth,
                child_node,
                decision=False,
            )

    def _create_child_node(
        self, X_feature_index, X_feature_threshold, branch_depth, squared_error, t
    ):
        # create and connect child node
        result = np.mean(t)
        child_node = Node(
            X_feature_index, X_feature_threshold, branch_depth, squared_error, result
        )

        return child_node

    def _add_child_node(self, parent_node, child_node, decision, branch_depth):
        # connect to parent node or make root
        if branch_depth == 0:
            self.root = child_node
        else:
            parent_node.add_child_node(child_node, decision)
            child_node.add_parent_node(parent_node)

    def _split_dataset(self, X, t, X_feature_index, X_feature_threshold):
        # the column we will split the dataset on
        X_feature = X[:, X_feature_index]

        # numeric columns are not deleted as they may be
        # used to create future decisions
        if not self._is_numeric(X_feature):
            X = np.delete(X, X_feature_index, axis=1)

        # get indices of rows where X_feature_index > X_feature_threshold
        yes_indices = np.asarray(X_feature >= X_feature_threshold).nonzero()[0]
        X_yes = X[yes_indices, :]
        t_yes = t[yes_indices]

        # get indices of rows where X_feature_index < X_feature_threshold
        no_indices = np.asarray(X_feature < X_feature_threshold).nonzero()[0]
        X_no = X[no_indices, :]
        t_no = t[no_indices]

        return X_yes, X_no, t_yes, t_no

    def _get_best_split(self, X, t):
        num_features = X.shape[1]
        num_inputs = X.shape[0]

        # initialize return values
        best_X_feature_index = None
        best_X_feature_threshold = None

        # initialize best_squared_error
        best_squared_error = np.inf

        # test all possible thresholds
        for X_feature_index in range(num_features):
            for X_feature_threshold_index in range(num_inputs):

                # input values before threshold for given feature
                X_before_threshold = X[:X_feature_threshold_index, X_feature_index]
                t_before_threshold = t[:X_feature_threshold_index]

                # input values after threshold for given feature
                X_after_threshold = X[X_feature_threshold_index:, X_feature_index]
                t_after_threshold = t[X_feature_threshold_index:]

                # predictions for both datasets
                prediction_before_threshold = np.mean(t_before_threshold)
                prediction_after_threshold = np.mean(t_after_threshold)

                # calculate squared_error
                squared_error_before = self._get_squared_error(
                    X_before_threshold, prediction_before_threshold
                )
                squared_error_after = self._get_squared_error(
                    X_after_threshold, prediction_after_threshold
                )

                squared_error = squared_error_before + squared_error_after

                # update return values
                if best_squared_error > squared_error:
                    best_squared_error = squared_error
                    best_X_feature_index = X_feature_index
                    best_X_feature_threshold = X[
                        X_feature_threshold_index, X_feature_index
                    ]

        return best_X_feature_index, best_X_feature_threshold, best_squared_error

    def _get_squared_error(self, X, regression_line):
        # calculate squared_error
        squared_error = 0
        for x in X:
            squared_error += (x - regression_line) ** 2

        return squared_error

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
