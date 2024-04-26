import sys

sys.path.insert(1, "..")
from DecisionTree import *
from utils import *
import numpy as np
import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class test_DecisionTree(unittest.TestCase):

    def setUp(self):
        mnist = load_breast_cancer(return_X_y=True)
        X_train, X_test, t_train, t_test = train_test_split(mnist[0], mnist[1])

        self.decisionTree = DecisionTree(max_depth=np.inf, classification_tree=True)

    def test_get_gini_index_for_columns(self):
        X_one = np.array([[1, 1, 0, 0, 1, 1, 0], [7, 12, 18, 35, 38, 50, 83]]).T
        t = np.array([0, 0, 1, 1, 1, 0, 0])

        expected_ginis = [0.405, 0.343]
        expected_thresholds = [0.5, 15]

        self.assertEqual(
            np.array_equal(
                np.round(self.decisionTree._get_gini_index_for_columns(X_one, t)[0], 3),
                expected_ginis,
            ),
            True,
        )
        self.assertEqual(
            np.array_equal(
                np.round(self.decisionTree._get_gini_index_for_columns(X_one, t)[1], 3),
                expected_thresholds,
            ),
            True,
        )

        X_two_numeric = np.array(
            [
                [7, 12, 18, 35, 38, 50, 83],
                [7, 12, 18, 35, 38, 50, 83],
                [7, 12, 18, 35, 38, 50, 83],
                [7, 12, 18, 35, 38, 50, 83],
                [7, 12, 18, 35, 38, 50, 83],
                [1, 0, 1, 1, 1, 0, 0],
                [7, 12, 18, 35, 38, 50, 83],
            ]
        ).T

        expected_ginis = [0.343, 0.343, 0.343, 0.343, 0.343, 0.214, 0.343]
        expected_thresholds = [15, 15, 15, 15, 15, 0.5, 15]

        self.assertEqual(
            np.array_equal(
                np.round(
                    self.decisionTree._get_gini_index_for_columns(X_two_numeric, t)[0],
                    3,
                ),
                expected_ginis,
            ),
            True,
        )
        self.assertEqual(
            np.array_equal(
                np.round(
                    self.decisionTree._get_gini_index_for_columns(X_two_numeric, t)[1],
                    3,
                ),
                expected_thresholds,
            ),
            True,
        )

        X_single_values = np.array([[1], [0], [1]]).T
        t_single_values = np.array([1, 1, 1])

        expected_ginis = [0,0,0]
        expected_thresholds = [0.5, 0.5, 0.5]

        self.assertEqual(
            np.array_equal(
                np.round(
                    self.decisionTree._get_gini_index_for_columns(X_single_values, t_single_values)[0],
                    3,
                ),
                expected_ginis,
            ),
            True
        )
        self.assertEqual(
            np.array_equal(
                np.round(
                    self.decisionTree._get_gini_index_for_columns(X_single_values, t_single_values)[1],
                    3,
                ),
                expected_thresholds,
            ),
            True,
        )



    def test_get_weighted_gini_index(self):

        # test for discrete and numeric columns for which we know gini index
        X_column_one = np.array([1, 1, 0, 0, 1, 1, 0])
        X_column_two = np.array([1, 0, 1, 1, 1, 0, 0])
        X_column_three = np.array([7, 12, 18, 35, 38, 50, 83])
        X_column_three = np.reshape(X_column_three, (-1, 1))

        t = np.array([0, 0, 1, 1, 1, 0, 0])

        expected_gini_one = 0.405
        expected_gini_two = 0.214
        expected_gini_three = 0.343

        self.assertEqual(
            round(self.decisionTree._get_weighted_gini_index(X_column_one, t), 3),
            expected_gini_one,
        )
        self.assertEqual(
            round(self.decisionTree._get_weighted_gini_index(X_column_two, t), 3),
            expected_gini_two,
        )
        self.assertEqual(
            round(
                self.decisionTree._get_gini_index_for_columns(X_column_three, t)[0][0],
                3,
            ),
            expected_gini_three,
        )

        # test edge cases where column only has one value
        X_column_edge_case_one = np.array([1])
        X_column_edge_case_two = np.array([0])
        t_edge_case = np.array([1])
        self.assertEqual(
            self.decisionTree._get_weighted_gini_index(
                X_column_edge_case_one, t_edge_case
            ),
            0,
        )
        self.assertEqual(
            self.decisionTree._get_weighted_gini_index(
                X_column_edge_case_two, t_edge_case
            ),
            0,
        )

        # test edge case where target only contains one value
        X_column_edge_case_three = np.array([1, 0, 0, 0])
        t_edge_case_three = np.array([1, 1, 1, 1])
        self.assertEqual(
            self.decisionTree._get_weighted_gini_index(
                X_column_edge_case_three, t_edge_case_three
            ),
            0,
        )

        # test same case but for numeric values
        X_column_edge_case_four = np.array([1, 0, 2, 9])
        X_column_edge_case_four = np.reshape(X_column_edge_case_four, (-1, 1))
        self.assertEqual(
            self.decisionTree._get_gini_index_for_columns(
                X_column_edge_case_four, t_edge_case_three
            )[0][0],
            0,
        )

    def test_is_numeric(self):

        # initialize discrete various cases
        X_discrete = np.array([1, 0, 0, 1])
        X_discrete_single = np.array([1])
        X_zeros = np.zeros((3, 1))
        X_ones = np.ones((6, 1))

        # test discrete cases
        self.assertFalse(self.decisionTree._is_numeric(X_discrete))
        self.assertFalse(self.decisionTree._is_numeric(X_discrete_single))
        self.assertFalse(self.decisionTree._is_numeric(X_zeros))
        self.assertFalse(self.decisionTree._is_numeric(X_ones))

        # initialize numeric cases
        X_numeric = np.array([9, 1, 0, 4])
        X_numeric_single = np.array([9])

        # test numeric cases
        self.assertTrue(self.decisionTree._is_numeric(X_numeric))
        self.assertTrue(self.decisionTree._is_numeric(X_numeric_single))

    def test_split_dataset(self):

        # discrete case
        X_zeros = np.zeros((3, 3), dtype=float)
        t = np.zeros((3,), dtype=float)
        X_yes, X_no, t_yes, t_no = self.decisionTree._split_dataset(X_zeros, t, 1, 0.5)

        self.assertEqual(X_yes.size, 0)
        self.assertEqual(t_yes.size, 0)
        self.assertEqual(X_no.shape, (3, 2))
        self.assertEqual(t_no.shape, (3,))

        X_ones = np.ones((3, 3), dtype=float)
        X_yes, X_no, t_yes, t_no = self.decisionTree._split_dataset(X_ones, t, 1, 0.5)

        self.assertEqual(X_no.size, 0)
        self.assertEqual(t_no.size, 0)
        self.assertEqual(X_yes.shape, (3, 2))
        self.assertEqual(t_yes.shape, (3,))

        X_ones_and_zeros = np.vstack((X_zeros, X_ones))
        X_column = np.array([[1, 0, 1, 1, 0, 1]]).T
        X_ones_and_zeros = np.hstack((X_ones_and_zeros, X_column))
        t = np.zeros((6,), dtype=float)
        X_yes, X_no, t_yes, t_no = self.decisionTree._split_dataset(
            X_ones_and_zeros, t, 3, 0.5
        )

        self.assertEqual(X_yes.shape, (4, 3))
        self.assertEqual(t_yes.shape, (4,))
        self.assertEqual(X_no.shape, (2, 3))
        self.assertEqual(t_no.shape, (2,))

        # numeric case

        X_numeric = np.array([[1, 0, 3], [1, 2, 4], [1, 4, 9], [1, 0, 1]])
        X_yes, X_no, t_yes, t_no = self.decisionTree._split_dataset(X_numeric, t, 1, 2)

        self.assertEqual(X_no.shape, (2, 3))
        self.assertEqual(t_no.shape, (2,))
        self.assertEqual(X_yes.shape, (2, 3))
        self.assertEqual(t_yes.shape, (2,))


if __name__ == "__main__":
    unittest.main()