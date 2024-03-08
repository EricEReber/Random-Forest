import numpy as np
from sklearn.utils import resample

class RandomForest:

    def __init__(self):
        self.list_subset_features = list()
        self.forest = list()
        self.num_trees = -1

    def fit(self, X, t, num_trees, num_features):
        self.num_trees = num_trees
        for n_tree in range(num_trees):
            # bootstrap X, t
            bootstrapped_X, bootstrapped_t = resample(X, t)

            # X is an input x features matrix 
            total_num_features = bootstrapped_X.shape[1]

            # use only subset of features
            subset_features = np.random.choice(total_num_features, num_features, replace=False)
            subset_bootstrapped_X = bootstrapped_X[:, subset_features]
            self.list_subset_features.append(subset_features)
            
            # fit decision_tree
            decision_tree = DecisionTree()
            decision_tree.fit(subset_bootstrapped_X, bootstrapped_t)
            self.forest.append(decision_tree)

    def predict(self, X):
        # check if ready to predict
        assert (self.num_trees != -1), "Please fit RandomForest first"

        list_predictions = list()
        for n_tree in range(self.num_trees):
            subset_features = self.list_subset_features[n_tree] 
            subset_X = X[:, subset_features]

            decision_tree = self.forest[n_tree] 
            prediction = decision_tree.predict(subset_X)
            list_predictions.append(prediction)

        # TODO actually use list_predictions


