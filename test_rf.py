from RandomForest import *

rf = RandomForest()
num_trees = 1
num_features = 2

X_train = np.array([[1,2,3, 4],[4,5,6,7],[7,8,9,1]]).T
print(f"{X_train=}")

z_train = np.array([1, 2, 3, 4]).T
print(f"{z_train=}")

rf.fit(X_train, z_train, num_trees, num_features)
rf.predict(X_train)
