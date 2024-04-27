class Node:
    def __init__(self, X_feature, threshold, result):
        self.X_feature = X_feature
        self.threshold = threshold
        self.t = result
        self.yes_child = None
        self.no_child = None

    def add_child_node(self, child_node, child_node_type):
        if child_node_type == "yes_node":
            self.yes_child = child_node
        else:
            self.no_child = child_node

    def has_children(self):
        if (self.yes_child == None) and (self.no_child == None):
            return False
        return True

    def get_X_feature(self):
        return self.X_feature

    def get_threshold(self):
        return self.threshold


    def get_yes_child(self):
        return self.yes_child

    def get_no_child(self):
        return self.no_child

    def add_result(self, t):
        self.t = t

    def get_result(self):
        return self.t

class ResultNode:
    def __init__(self, t):
        self.t = t

    def get_result(self):
        return self.t
