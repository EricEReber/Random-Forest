import sys

sys.path.insert(1, "..")
from DecisionTreeRegressor import *
from utils import *
import numpy as np
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class test_DecisionTreeRegressor(unittest.TestCase):

    def setUp(self):
        self.dtRegressor = DecisionTreeRegressor(min_inputs=2)

    def test_create_child_node(self):
        # see if all variables of child node are added correctly
        expected_index = 0
        expected_threshold = 1
        t = [10, 20, 30, 40, 50]
        expected_result = np.mean(t)

        child_node = self.dtRegressor._create_child_node(expected_index, expected_threshold, t)
        self.assertEqual(child_node.get_X_feature_index(), expected_index)
        self.assertEqual(child_node.get_X_feature_threshold(), expected_threshold)
        self.assertEqual(child_node.get_regression_result(), expected_result)


    def test_add_child_node(self):
        # check for adding root
        expected_index = 0
        expected_threshold = 1
        expected_result = 3
        parent_node = Node(1, 1, 1)
        child_node = Node(expected_index, expected_threshold, expected_result)
        branch_depth = 0
        decision = True

        self.dtRegressor._add_child_node(parent_node, child_node, decision, branch_depth)
        self.assertEqual(self.dtRegressor.root.get_X_feature_index(), expected_index)
        self.assertEqual(self.dtRegressor.root.get_X_feature_threshold(), expected_threshold)
        self.assertEqual(self.dtRegressor.root.get_regression_result(), expected_result)

        # check for adding child to parent node
        branch_depth = 1
        decision = False

        self.dtRegressor._add_child_node(self.dtRegressor.root, child_node, decision, branch_depth)
        self.assertEqual(self.dtRegressor.root.get_X_feature_index(), expected_index)
        self.assertEqual(self.dtRegressor.root.get_X_feature_threshold(), expected_threshold)
        self.assertEqual(self.dtRegressor.root.get_regression_result(), expected_result)

    def test_split_dataset(self):
        # discrete case
        X_zeros = np.zeros((3, 3), dtype=float)
        t = np.zeros((3,), dtype=float)
        X_yes, X_no, t_yes, t_no = self.dtRegressor._split_dataset(X_zeros, t, 1, 0.5)

        self.assertEqual(X_yes.size, 0)
        self.assertEqual(t_yes.size, 0)
        self.assertEqual(X_no.shape, (3, 2))

        X_ones = np.ones((3, 3), dtype=float)
        X_yes, X_no, t_yes, t_no = self.dtRegressor._split_dataset(X_ones, t, 1, 0.5)

        self.assertEqual(X_no.size, 0)
        self.assertEqual(t_no.size, 0)
        self.assertEqual(t_no.shape, (3,))
        self.assertEqual(X_yes.shape, (3, 2))
        self.assertEqual(t_yes.shape, (3,))

        X_ones_and_zeros = np.vstack((X_zeros, X_ones))
        X_column = np.array([[1, 0, 1, 1, 0, 1]]).T
        X_ones_and_zeros = np.hstack((X_ones_and_zeros, X_column))
        t = np.zeros((6,), dtype=float)
        X_yes, X_no, t_yes, t_no = self.dtRegressor._split_dataset(
            X_ones_and_zeros, t, 3, 0.5
        )

        self.assertEqual(X_yes.shape, (4, 3))
        self.assertEqual(t_yes.shape, (4,))
        self.assertEqual(X_no.shape, (2, 3))
        self.assertEqual(t_no.shape, (2,))

        # numeric case

        X_numeric = np.array([[1, 0, 3], [1, 2, 4], [1, 4, 9], [1, 0, 1]])
        X_yes, X_no, t_yes, t_no = self.dtRegressor._split_dataset(X_numeric, t, 1, 2)

        self.assertEqual(X_no.shape, (2, 3))
        self.assertEqual(t_no.shape, (2,))
        self.assertEqual(X_yes.shape, (2, 3))
        self.assertEqual(t_yes.shape, (2,))

        # additional tests

        X = np.array([[1, 1, 0, 0, 1, 1, 0], [7, 12, 18, 35, 38, 50, 83]]).T
        t = np.array([0, 0, 1, 1, 1, 0, 0])
        discrete_threshold = 0.5
        discrete_index = 0

        X_yes, X_no, t_yes, t_no = self.dtRegressor._split_dataset(
            X, t, discrete_index, discrete_threshold
        )

        self.assertEqual(X_yes.shape, (4, 1))
        self.assertEqual(t_yes.shape, (4,))
        self.assertEqual(X_no.shape, (3, 1))
        self.assertEqual(t_no.shape, (3,))

        numeric_threshold = 12
        numeric_index = 1

        X_yes, X_no, t_yes, t_no = self.dtRegressor._split_dataset(
            X, t, numeric_index, numeric_threshold
        )

        self.assertEqual(X_yes.shape, (6, 2))
        self.assertEqual(t_yes.shape, (6,))
        self.assertEqual(X_no.shape, (1, 2))
        self.assertEqual(t_no.shape, (1,))

    def test_get_best_split(self):
        # test if chooses best index, threshold
        X = np.array([[1, 1, 1, 10, 11, 11], [5, 4, 6, 5, 4, 5], [3, 3, 3, 3, 3, 3]]).T
        t = np.array([10, 10, 10, 20, 20, 20])

        expected_best_index = 0
        expected_best_threshold = 10

        best_index, best_threshold = self.dtRegressor._get_best_split(X, t)
        self.assertEqual(best_index, expected_best_index)
        self.assertEqual(best_threshold, expected_best_threshold)

        # edge case with one entry
        X = np.array([[10]])
        t = np.array([100])

        expected_best_index = 0
        expected_best_threshold = 10

        best_index, best_threshold = self.dtRegressor._get_best_split(X, t)
        self.assertEqual(best_index, expected_best_index)
        self.assertEqual(best_threshold, expected_best_threshold)

        # edge case with empty array
        X = np.array([[]])
        t = np.array([])

        expected_best_index = None
        expected_best_threshold = None

        best_index, best_threshold = self.dtRegressor._get_best_split(X, t)
        self.assertEqual(best_index, expected_best_index)
        self.assertEqual(best_threshold, expected_best_threshold)

    def test_get_squared_error(self):
        # test if calculation is correct
        X = [4, -4, 4]
        regression_line = 0

        expected_result = 16 + 16 + 16
        self.assertEqual(
            self.dtRegressor._get_squared_error(X, regression_line), expected_result
        )

        # test second case
        X = [0, -4, 4]
        regression_line = 2

        expected_result = 4 + 36 + 4
        self.assertEqual(
            self.dtRegressor._get_squared_error(X, regression_line), expected_result
        )

    def test_is_numeric(self):

        # initialize discrete various cases
        X_discrete = np.array([1, 0, 0, 1])
        X_discrete_single = np.array([1])
        X_zeros = np.zeros((3, 1))
        X_ones = np.ones((6, 1))

        # test discrete cases
        self.assertFalse(self.dtRegressor._is_numeric(X_discrete))
        self.assertFalse(self.dtRegressor._is_numeric(X_discrete_single))
        self.assertFalse(self.dtRegressor._is_numeric(X_zeros))
        self.assertFalse(self.dtRegressor._is_numeric(X_ones))

        # initialize numeric cases
        X_numeric = np.array([9, 1, 0, 4])
        X_numeric_single = np.array([9])

        # test numeric cases
        self.assertTrue(self.dtRegressor._is_numeric(X_numeric))
        self.assertTrue(self.dtRegressor._is_numeric(X_numeric_single))

if __name__ == "__main__":
    unittest.main()
