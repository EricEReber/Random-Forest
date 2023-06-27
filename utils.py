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


def bootstrap(X_train, z_train):
    bootstrapped_X_train = 
