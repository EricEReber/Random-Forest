import numpy as np
from DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.utils import resample


class RandomForestClassifier:

    def __init__(self):
        self.list_subset_features = list()
        self.forest = list()

    def fit(self, X, t, num_trees, num_features, max_depth=np.inf):
        for n_tree in range(num_trees):
            # bootstrap X, t
            bootstrapped_X, bootstrapped_t = resample(X, t)

            # X is an input x features matrix
            total_num_features = bootstrapped_X.shape[1]

            # use only subset of features
            subset_features = np.random.choice(
                total_num_features, num_features, replace=False
            )
            subset_bootstrapped_X = bootstrapped_X[:, subset_features]
            self.list_subset_features.append(subset_features)

            # fit decision_tree
            decision_tree = DecisionTreeClassifier(max_depth=max_depth)
            decision_tree.fit(subset_bootstrapped_X, bootstrapped_t)
            self.forest.append(decision_tree)

    def predict(self, X):
        num_preds = X.shape[0]
        num_trees = len(self.forest)
        tree_predictions = np.zeros((num_preds, num_trees), dtype=int)
        for n_tree in range(num_trees):
            # use the subset of features corresponding to each tree
            subset_features = self.list_subset_features[n_tree]
            subset_X = X[:, subset_features]

            # get the decision tree
            decision_tree = self.forest[n_tree]

            # predict
            prediction = decision_tree.predict(subset_X)
            tree_predictions[:, n_tree] = prediction

        # get most occuring predicted class
        prediction = np.zeros((num_preds))
        for pred in range(num_preds):
            prediction[pred] = np.bincount(tree_predictions[pred, :]).argmax()

        return prediction
