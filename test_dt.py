from DecisionTree import *
import numpy as np

dt = DecisionTree()

X = np.array([[1,0,0],[0,1,0],[0,0,1]])
t = np.array([1,1,0])

w = dt._get_weighted_gini_index(X[:,0], t)
print(w)
