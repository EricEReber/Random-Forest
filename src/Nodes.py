class Node:
    # TODO split into RegressorNode and ClassifierNode, but first check what is needed for classifination
    def __init__(
        self, X_feature_index, X_feature_threshold, result, node_depth=0, squared_error=0
    ):
        self.X_feature_index = X_feature_index
        self.X_feature_threshold = X_feature_threshold
        self.node_depth = node_depth
        self.t = result
        self.squared_error = squared_error
        self.parent = None
        self.yes_child = None
        self.no_child = None

    def add_child_node(self, child_node, decision):
        if decision:
            self.yes_child = child_node
        else:
            self.no_child = child_node

    def remove_children(self):
        self.yes_child = None
        self.no_child = None

    def add_parent_node(self, parent_node):
        self.parent = parent_node

    def has_children(self):
        if (self.yes_child == None) and (self.no_child == None):
            return False
        return True

    def get_X_feature_index(self):
        return self.X_feature_index

    def get_X_feature_threshold(self):
        return self.X_feature_threshold

    def get_yes_child(self):
        return self.yes_child

    def get_no_child(self):
        return self.no_child

    def get_parent(self):
        return self.parent

    def get_node_depth(self):
        return self.node_depth

    def get_squared_error(self):
        return self.squared_error

    def get_classification_result(self, X_feature):
        if X_feature >= self.X_feature_threshold:
            return self.t
        else:
            return not self.t

    def get_regression_result(self):
        return self.t
