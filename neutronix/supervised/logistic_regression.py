import numpy as np

class LogisticRegression:
    def __init__(self):
        self.optimizer = None
        self.parameters = None
        self.weights = None
    
    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, x):
        return self.sigmoid(x @ self.parameters)

    def loss(self, X_train, Y_train):
        y_pred = np.clip(self.hypothesis(X_train), 1e-15, 1 - 1e-15)
        return -np.sum(Y_train * np.log(y_pred) + (1- Y_train) * np.log(1- y_pred))
    
    def gradient(self, X_train, Y_train):
        return X_train.T @ (self.hypothesis(X_train) - Y_train)
    
    def hessian(self, X_train):
        p = self.hypothesis(X_train)
        r = p * (1-p)
        return X_train.T @ (r * X_train)
    
    def newtons_method(self, X_train, Y_train):
        return -np.linalg.solve(self.hessian(X_train), self.gradient(X_train , Y_train))
    
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
            if self.optimizer == self.newtons_method:
                self.parameters += self.newtons_method(X_train, Y_train)
            else:
                self.parameters+=self.optimizer(X_train, Y_train, self.hypothesis(X_train))
            if e % 1000 == 0 and verbose:
                print(f"Epoch {e:4d} | Loss: {self.loss(X_train , Y_train):.4f}")
    
    def predict(self, X_test, probability=False):
        if self.parameters is None:
            raise Exception("Train The Model First")
        if (X_test.ndim == 1):
            X_test = X_test.reshape(-1, 1)
        Y_pred_prob = self.hypothesis(np.hstack([np.ones((X_test.shape[0], 1)), X_test]))
        if probability:
            return Y_pred_prob
        return (Y_pred_prob >= 0.5).astype(int)
    
    def score(self, X_test, Y_test):
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1,1)
        Y_pred = self.predict(X_test, probability=False)
        return np.mean(Y_pred == Y_test)
    
    def save(self, directory='model.npy'):
        np.save(directory, self.parameters)

    def load(self, directory='model.npy'):
        self.parameters = np.load(directory)

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from optimizers.gradient_descent import GradientDescent

    X = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=float)
    Y = np.array([0, 0, 0, 1, 1, 1], dtype=float)

    model = LogisticRegression()
    model.optimizer = GradientDescent(alpha=0.1).batch_gradient_descent
    model.fit(X, Y, epoch=10000, verbose=False)

    print(model.predict(np.array([1.0, 2.5])))
    print(model.score(X, Y))
