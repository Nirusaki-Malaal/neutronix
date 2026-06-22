import numpy as np


class LWLR:
    def __init__(self, bandwidth=0.5, query=None):
        self.optimizer = None
        self.parameters = None
        self.bandwidth = bandwidth
        self.query = query
        self.weight = None

    def weight_matrix(self, X_train):
        if self.query is None:
            raise Exception("[ERROR] Query Point is Not Configured")
        return np.exp((-np.sum((X_train - self.query) ** 2, axis=1)) / (2 * (self.bandwidth ** 2)))

    def hypothesis(self, X_train):
        return X_train @ self.parameters

    def loss(self, X_train, Y_train):
        if self.weight is None:
            self.weight = self.weight_matrix(X_train)
        return np.sum(self.weight.reshape(-1, 1) * (self.hypothesis(X_train) - Y_train) ** 2) / 2

    def closed_form(self, X_train, Y_train):
        if self.parameters is None:
            raise Exception("[ERROR] Parameters are not initalized")
        if self.weight is None:
            self.weight = self.weight_matrix(X_train)
        W = np.diag(self.weight)
        return np.linalg.pinv(X_train.T @ W @ X_train) @ X_train.T @ W @ Y_train

    def fit(self, X_train, Y_train, epoch=10, verbose=True):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)

        m, n = X_train.shape
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)
        if np.ndim(self.query) == 0:
            self.query = np.array([[self.query]])
        if self.query.ndim == 1:
            self.query = self.query.reshape(1, -1)

        X_train = np.hstack([np.ones((m, 1)), X_train])
        self.query = np.hstack([np.ones((self.query.shape[0], 1)), self.query])
        m, n = X_train.shape

        if self.parameters is None:
            self.parameters = np.zeros((n, 1))
        if self.optimizer != self.closed_form:
            for e in range(epoch):
                if self.optimizer is None:
                    raise Exception("INITIALIZE THE OPTIMIZER FIRST")

                self.parameters += self.optimizer(X_train, Y_train, self.hypothesis(X_train))

                if e % 1000 == 0 and verbose:
                    print(f"Epoch {e:4d} | Loss: {self.loss(X_train, Y_train):.4f}")
        else:
            self.parameters = self.optimizer(X_train, Y_train)
            if verbose:
                print(f"Loss: {self.loss(X_train, Y_train):.4f}")

    def predict(self, X_test):
        if self.parameters is None:
            raise Exception("Train The Model First")
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        return self.hypothesis(np.hstack([np.ones((X_test.shape[0], 1)), X_test]))

    def save(self, directory='model.npy'):
        np.save(directory, self.parameters)

    def load(self, directory='model.npy'):
        self.parameters = np.load(directory)


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from optimizers.gradient_descent import GradientDescent
    feature = np.array([
        -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0,
        0.5, 1.0, 1.5, 2.0, 2.5, 3.0
    ], dtype=float)
    target = np.array([
        -0.14, -0.60, -0.91, -0.99, -0.84, -0.48, 0.02,
        0.51, 0.84, 0.99, 0.91, 0.60, 0.14
    ], dtype=float)

    model = LWLR(bandwidth=0.5, query=1.0)
    model.optimizer = model.closed_form
    model.fit(feature, target, verbose=True)
    print(model.predict(np.array([1.0]))[0][0])
