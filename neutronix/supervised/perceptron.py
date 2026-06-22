import numpy as np

class Perceptron:
    def __init__(self):
        self.optimizer = None
        self.parameters = None
        self.weights = None

    def hypothesis(self, x):
        return ((x @ self.parameters) >= 0).astype(int)

    def loss(self, X_train, Y_train):
        return np.sum((self.hypothesis(X_train) - Y_train)**2) / 2
    
    def fit(self,X_train, Y_train,epoch=10, verbose=True):
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)
        
        m,n = X_train.shape
        if (Y_train.ndim == 1):
            Y_train = Y_train.reshape(-1, 1)

        X_train = np.hstack([np.ones((m, 1)), X_train])
        m,n = X_train.shape

        if self.parameters is None:
            self.parameters = np.zeros((n,1))
        for e in range(epoch):
            if self.optimizer is None:
                raise Exception("INITIALIZE THE OPTIMIZER FIRST")
            self.parameters+=self.optimizer(X_train, Y_train, self.hypothesis(X_train))
            if e % 1000 == 0 and verbose:
                print(f"Epoch {e:4d} | Loss: {self.loss(X_train , Y_train):.4f}")
    
    def predict(self, X_test, probability=False):
        if self.parameters is None:
            raise Exception("Train The Model First")
        if (X_test.ndim == 1):
            X_test = X_test.reshape(-1, 1)
        Y_pred = self.hypothesis(np.hstack([np.ones((X_test.shape[0], 1)), X_test]))
        return Y_pred
    
    def score(self, X_test, Y_test):
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1,1)
        Y_pred = self.predict(X_test)
        return np.mean(Y_pred == Y_test)
    
    def save(self, directory='model.npy'):
        np.save(directory, self.parameters)

    def load(self, directory='model.npy'):
        self.parameters = np.load(directory)

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from optimizers.gradient_descent import GradientDescent
    X = np.array([0, 1, 2, 3], dtype=float)
    Y = np.array([0, 0, 1, 1])
    model = Perceptron()
    model.optimizer = GradientDescent(alpha=0.1).batch_gradient_descent
    model.fit(X, Y, epoch=100, verbose=False)
    print(model.predict(X))
    print(model.score(X, Y))
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)
    Y = np.array([0, 0, 0, 1])
    model = Perceptron()
    model.optimizer = GradientDescent(alpha=0.1).batch_gradient_descent
    model.fit(X, Y, epoch=100, verbose=False)
    print(model.predict(X))
    print(model.score(X, Y))
