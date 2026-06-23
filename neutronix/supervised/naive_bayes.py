import numpy as np

# NAIVE BAYES IS IMPLEMENTED WITH A BUILTIN OPTIMIZER FUNCTION

class NaiveBayes:
    def __init__(self):
        self.phi = None # CLASS PRIOR
        self.phi_0 = None
        self.phi_1 = None
        self.num_values = None
        self.classes = None

    def optimizer(self,X_train,Y_train):
        m,n = X_train.shape
        total_y1 = np.sum(Y_train == 1)
        total_y0 = np.sum(Y_train == 0)
        self.num_values = np.max(X_train, axis=0).astype(int) + 1
        self.phi_0 = [np.zeros(v) for v in self.num_values]
        self.phi_1 = [np.zeros(v) for v in self.num_values]
        self.phi = (np.sum(Y_train)+1)/(m+2)
        
        for j in range(n):
            values = self.num_values[j] 
            counts_y1 = np.zeros(values)
            counts_y0 = np.zeros(values)
            for i in range(m):
                val = X_train[i][j]
                if Y_train[i] == 1:
                    counts_y1[val] +=1
                else:
                    counts_y0[val] +=1
            self.phi_0[j] = (counts_y0 + 1) / (total_y0 + values) # LAPLACE SMOOTHING 
            self.phi_1[j] = (counts_y1 + 1) / (total_y1 + values)
    
    def fit(self,X_train, Y_train):
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)
        
        m,n = X_train.shape

        if (Y_train.ndim == 1):
            Y_train = Y_train.reshape(-1, 1)

        self.num_classes = np.unique(Y_train.flatten()).shape[0]

        self.optimizer(X_train , Y_train.flatten())

    def predict(self, X_test):
        if self.phi is None:
            raise Exception("Train The Model First")
        if (X_test.ndim == 1):
            X_test = X_test.reshape(1, -1)
        pred = []
        for x in X_test.astype(int):
            p1, p0 = np.log(self.phi), np.log(1 - self.phi)
            for j, val in enumerate(x):
                p1 += np.log(self.phi_1[j][val])
                p0 += np.log(self.phi_0[j][val])
            pred.append(p1 > p0)
        return np.asarray(pred, dtype=int)
    
    
    def score(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        return np.mean(Y_pred == np.asarray(Y_test).flatten())
    
    def save(self, directory='model.npy'):
        np.savez(directory, phi=self.phi, phi_0=self.phi_0, phi_1=self.phi_1)

    def load(self, directory='model.npy'):
        data = np.load(directory, allow_pickle=True)
        self.phi = data["phi"] 
        self.phi_0 = data["phi_0"] 
        self.phi_1 = data["phi_1"] 
