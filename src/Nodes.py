class Node:
    def __init__(self, X_feature_index, X_feature_threshold, result):
        self.X_feature_index = X_feature_index
        self.X_feature_threshold = X_feature_threshold
        self.t = result
        self.yes_child = None
        self.no_child = None

    def add_child_node(self, child_node, decision):
        if decision: 
            self.yes_child = child_node
        else:
            self.no_child = child_node

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

    def get_classification_result(self, X_feature):
        if X_feature >= self.X_feature_threshold:
            return self.t
        else:
            return not self.t

    def get_regression_result(self):
        return self.t
