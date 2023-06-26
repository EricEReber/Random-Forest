import numpy as np
from sklearn.utils import resample

class RandomForest:

    def __init__(self):
        pass

    def fit(self, X, t, num_trees, num_features):
        forest = list()
        for n in range(num_trees):
            # bootstrap X, t
            bootstrapped_X, bootstrapped_t = resample(X, t)
            print(f"{bootstrapped_X=}")
            print(f"{bootstrapped_t=}")


            # X is an input x features matrix 
            total_num_features = bootstrapped_X.shape[1]

            # use only subset of features
            # NOTE representation might get messed up because columns get scrambled
            # NOTE might be better to zero-out columns instead of removing and placing around
            bootstrapped_X = bootstrapped_X[:, np.random.choice(total_num_features, num_features, replace=False)]
            
            print(f"{bootstrapped_X=}")
            # fit tree
            # tree = Tree()
            # tree.fit(bootstrapped_X, bootstrapped_t)
            # forest.add(tree)

    def predict(self):
        pass
