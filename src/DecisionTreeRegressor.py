import numpy as np
from Nodes import Node


class DecisionTreeRegressor:
    def __init__(self, max_depth=np.inf, min_inputs=0, alpha=0):
        self.max_depth = max_depth
        self.min_inputs = min_inputs
        self.alpha = alpha
        self.root = None
        self.total_squared_error = 0
        self.leaf_nodes = set()
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

    def calculate_tree_score(self):
        self.tree_score = self.total_squared_error + self.alpha * len(self.leaf_nodes)
        return self.tree_score

    def get_tree_depth(self):
        # get highest tree depth
        tree_depth = 0
        for leaf_node in self.leaf_nodes:
            if leaf_node.get_node_depth() > tree_depth:
                tree_depth = leaf_node.get_node_depth()

        return tree_depth

    def prune(self):
        tree_depth = self.get_tree_depth()

        # edge case where we'd prune root since root has no parent
        if tree_depth == 0:
            return tree_depth

        # remove leaf_nodes with highest depth
        old_leaf_nodes = set()
        new_leaf_nodes = set()
        for leaf_node in self.leaf_nodes:
            if leaf_node.get_node_depth() == tree_depth:
                # remove parent's children
                parent_node = leaf_node.get_parent()
                parent_node.remove_children()

                # add parents to leaf_nodes
                new_leaf_nodes.add(parent_node)

                # remove leaf node from leaf_nodes
                old_leaf_nodes.add(leaf_node)

                # subtract squared_error
                self.total_squared_error -= leaf_node.get_squared_error()

        self.leaf_nodes = self.leaf_nodes - old_leaf_nodes
        self.leaf_nodes.update(new_leaf_nodes)

    def predict(self, X):
        num_features = X.shape[1]
        num_inputs = X.shape[0]
        # initialize output vector
        prediction = np.zeros((num_inputs))

        # for each test input
        for i in range(num_inputs):
            # start at root
            current_node = self.root
            next_node = current_node

            # for keeping track of global X_feature_index values
            feature_indices = [j for j in range(num_features)]
            is_numeric_feature_index = [self._check_if_numeric_feature(X[:, i]) for i in range(num_features)]

            # traverse down tree until leaf node
            while next_node is not None:
                current_node = next_node
                # select index relative to removed indices
                X_feature_index = feature_indices[current_node.get_X_feature_index()]
                print(f"{is_numeric_feature_index=}")
                print(f"{feature_indices=}")
                print(f"{X_feature_index=}")
                # by removing incides of discrete features,
                # we can keep track of what index corresponds to what features,
                # as discrete columns are removed during _split_dataset()
                if not is_numeric_feature_index[X_feature_index]:
                    feature_indices.pop(X_feature_index)
                    is_numeric_feature_index.pop(X_feature_index)

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
        self.total_squared_error += squared_error

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
            X_feature_index,
            X_feature_threshold,
            result,
            node_depth=branch_depth,
            squared_error=squared_error,
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
        if not self._check_if_numeric_feature(X_feature):
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

    def _check_if_numeric_feature(self, X_feature):
        # assumes no == 0, yes == 1 in discrete case
        if (
            np.array_equal(np.unique(X_feature), np.array([0, 1]))
            or np.array_equal(np.unique(X_feature), np.array([0]))
            or np.array_equal(np.unique(X_feature), np.array([1]))
        ):
            return False
        else:
            return True
