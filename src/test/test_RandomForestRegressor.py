import sys

sys.path.insert(1, "..")
from RandomForestRegressor import *
from utils import *
import numpy as np
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class test_RandomForestRegressor(unittest.TestCase):
    def setUp(self):
        mnist = load_breast_cancer(return_X_y=True)
        self.X_train, self.X_test, self.t_train, self.t_test = train_test_split(
            mnist[0], mnist[1]
        )

        self.rf_regressor = RandomForestRegressor()

    def test_fit(self):
        X = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        t = np.array([1, 1, 0])
        NUM_TREES = 5
        NUM_FEATURES = 2
        MAX_DEPTH = 5
        self.rf_regressor.fit(X, t, NUM_TREES, NUM_FEATURES, MAX_DEPTH)

        self.assertEqual(len(self.rf_regressor.forest), NUM_TREES)
        self.assertEqual(np.array(self.rf_regressor.subset_features).shape, (NUM_TREES, NUM_FEATURES))

    def test_predict(self):
        NUM_TREES = 10
        NUM_FEATURES = 4
        MAX_DEPTH = 9
        X = np.array(
            [[1, 1, 0, 0, 1], [1, 1, 0, 0, 1], [0, 0, 1, 0, 0], [1, 1, 0, 0, 1]]
        ).T
        t = np.array([1, 1, 0, 0, 1])

        NUM_FEATURES = 3
        self.rf_regressor.fit(X, t, NUM_TREES, NUM_FEATURES, MAX_DEPTH)

        self.assertEqual(self.rf_regressor.predict(X).shape, t.shape)

if __name__ == "__main__":
    unittest.main()
