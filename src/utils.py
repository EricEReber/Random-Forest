import numpy as np

def bootstrap(
    X_train: np.ndarray,
    z_train: np.ndarray,
    bootstraps: int,
):
    bootstrapped_tuples = list()

    for i in range(bootstraps):
        X_, z_ = resample(X_train, z_train)
        bootstrapped_tuples.append((X_, z_))

    return bootstrapped_tuples

def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model) ** 2) / n
