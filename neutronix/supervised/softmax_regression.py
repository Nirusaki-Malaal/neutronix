import numpy as np

class SoftmaxRegression:
    def __init__(self):
        self.optimizer = None
        self.parameters = None
        self.classes = None
        
    def hypothesis(self, X_train):
        z = X_train @ self.parameters
        z = z - np.max(z, axis=1, keepdims=True) ## overflow check
        z = np.exp(z)
        return z / np.sum(z, axis=1, keepdims=True)

    def loss(self, X_train, Y_train):
        y_hat = self.hypothesis(X_train)
        enc = self.encoding(X_train, Y_train)
        return -np.sum(enc * np.log(y_hat + 1e-9)) / X_train.shape[0]
    
    def encoding(self,X_train, Y_train): # one hot encoding
        Y_train = Y_train.flatten()
        Y_train = np.searchsorted(self.classes, Y_train)
        num_classes = self.parameters.shape[1]
        enc = np.zeros((X_train.shape[0], num_classes))
        enc[np.arange(X_train.shape[0]), Y_train] = 1 # np.arange(5) = [0 , 1 , 2 ,3 ,4]
        return enc
    
    def fit(self,X_train, Y_train,epoch=10, verbose=True):
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)
        
        m,n = X_train.shape
        if (Y_train.ndim == 1):
            Y_train = Y_train.reshape(-1, 1)
        self.classes = np.unique(Y_train.flatten())
        num_classes = len(self.classes)

        X_train = np.hstack([np.ones((m, 1)), X_train])
        m,n = X_train.shape

        if self.parameters is None or self.parameters.shape != (n, num_classes):
            self.parameters = np.zeros((n, num_classes))
        for e in range(epoch):

            if self.optimizer is None:
                raise Exception("INITIALIZE THE OPTIMIZER FIRST")
                
            self.parameters+=self.optimizer(X_train, self.encoding(X_train, Y_train), self.hypothesis(X_train))

            if e % 1000 == 0 and verbose:
                print(f"Epoch {e:4d} | Loss: {self.loss(X_train , Y_train):.4f}")
    
    def predict(self, X_test):
        if self.parameters is None:
            raise Exception("Train The Model First")
        if (X_test.ndim == 1):
            X_test = X_test.reshape(-1, 1)
        Y_pred = np.argmax(self.hypothesis(np.hstack([np.ones((X_test.shape[0], 1)), X_test])), axis=1).flatten()
        if self.classes is None:
            return Y_pred
        return self.classes[Y_pred]
    
    def score(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        return np.mean(Y_pred == np.asarray(Y_test).flatten())
    
    def save(self, directory='model.npy'):
        np.save(directory, {"parameters": self.parameters, "classes": self.classes})

    def load(self, directory='model.npy'):
        data = np.load(directory, allow_pickle=True)
        if data.shape == ():
            data = data.item()
            self.parameters = data["parameters"]
            self.classes = data["classes"]
        else:
            self.parameters = data
            self.classes = np.arange(self.parameters.shape[1])
