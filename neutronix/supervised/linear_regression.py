import numpy as np


class LinearRegression:
    def __init__(self):
        self.optimizer = None
        self.parameters = None

    def hypothesis(self, x):
        return  x @ self.parameters

    def loss(self, X_train, Y_train):
        return np.sum((self.hypothesis(X_train) - Y_train)**2) / 2
    
    def closed_form(self, X_train, Y_train):
        if self.parameters is None:
            raise Exception("[ERROR] Parameters are not initalized")
        return np.linalg.inv(X_train.T @ X_train) @ X_train.T @ Y_train
    
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
        if self.optimizer != self.closed_form:
            for e in range(epoch):

                if self.optimizer is None:
                    raise Exception("INITIALIZE THE OPTIMIZER FIRST")

                self.parameters+=self.optimizer(X_train, Y_train, self.hypothesis(X_train))

                if e % 1000 == 0 and verbose:
                    print(f"Epoch {e:4d} | Loss: {self.loss(X_train , Y_train):.4f}")
        else:
            self.parameters = self.optimizer(X_train, Y_train)
            if verbose:
                print(f"Loss: {self.loss(X_train , Y_train):.4f}")
    
    def predict(self, X_test):
        if self.parameters is None:
            raise Exception("Train The Model First")
        if (X_test.ndim == 1):
            X_test = X_test.reshape(-1, 1)
        return self.hypothesis(np.hstack([np.ones((X_test.shape[0], 1)), X_test]))
    
    def score(self, X_test, Y_test):
        if Y_test.ndim == 1:
            Y_test = Y_test.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1,1)
        Y_pred = self.predict(X_test)
        ss_res = np.sum((Y_test - Y_pred) ** 2)
        ss_tot = np.sum((Y_test - np.mean(Y_test)) ** 2)
        return 1 - (ss_res / ss_tot)

    
    def save(self, directory='model.npy'):
        np.save(directory, self.parameters)

    def load(self, directory='model.npy'):
        self.parameters = np.load(directory)

if __name__ == "__main__":
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from optimizers.gradient_descent import GradientDescent
    price = np.array([
    400, 232, 318, 450, 280, 375, 500, 195, 420, 360,
    275, 415, 330, 250, 490, 210, 385, 445, 305, 260,
    520, 175, 395, 465, 315, 240, 410, 355, 285, 430, 
    180, 370, 495, 225, 340, 460, 295, 405, 265, 480,
    345, 215, 390, 435, 270, 510, 300, 365, 245, 420
], dtype=float) ## y
    size = np.array([
    2104, 1416, 1534, 2400, 1300, 1800, 2650, 1100, 2200, 1750,
    1250, 2180, 1600, 1180, 2580, 1050, 1900, 2350, 1480, 1220,
    2750, 980,  2000, 2480, 1560, 1120, 2150, 1720, 1350, 2280,
    1000, 1820, 2600, 1080, 1650, 2420, 1420, 2100, 1260, 2520,
    1680, 1060, 1960, 2300, 1290, 2700, 1460, 1780, 1140, 2200
], dtype=float) ## x
    size = (size - size.mean()) / size.std()
    model = LinearRegression()
    model.optimizer = model.closed_form
    model.fit(size , price, epoch=100000, verbose=True)
    model.save()
    #model.load()
    X_test = np.array([
    1150,
    1400,
    1700,
    1950,
    2250,
    2500
    ], dtype=float)
    X_test = (X_test - X_test.mean()) / X_test.std()
    Y_test = np.array([
    245,
    295,
    355,
    390,
    430,
    485
], dtype=float)
    print(model.score(X_test , Y_test))
