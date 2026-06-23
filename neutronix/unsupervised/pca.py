import numpy as np
class PCA():
    def __init__(self, K=10):
        self.X = None
        self.mu = None
        self.std = None
        self.K = K
        self.U = None

    def fit(self, X_train):
        X_train = np.array(X_train)
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)
        self.X = np.array(X_train)
        self.mu = np.sum(self.X , axis=0) / X_train.shape[0]
        self.X = self.X - self.mu
        self.std = np.std(self.X, axis=0)
        self.std[self.std == 0] = 1
        self.X = self.X / self.std
        self.optimizer()

    def optimizer(self):
        if self.X is None:
            raise Exception("Train The Model First")
        sigma  = (self.X.T @ self.X) / self.X.shape[0]
        eigenvalues, eigenvectors = np.linalg.eigh(sigma) ## ascending
        eigenvalues = eigenvalues[::-1] ## descending 
        eigenvectors = eigenvectors [:, ::-1] # (n,n) # pick top K [u1 , u2 ..... un]
        self.U = eigenvectors[: , :self.K] # (n,k)
    
    def transform(self, x): # x is an example
        if self.U is None:
            raise Exception("Train The Model First")
        return (np.array(x) - self.mu) / self.std @ self.U[:, :self.K]
    
    def reconstruction(self,y):
        if self.U is None:
            raise Exception("Train The Model First")
        return ((y @ self.U.T) * self.std) + self.mu
    
    def reconstruction_error(self, X_test):
        X_test = np.array(X_test)
        if (X_test.ndim == 1):
            X_test = X_test.reshape(1, -1)
        Y_pred = self.transform(X_test)
        return np.linalg.norm(X_test - self.reconstruction(Y_pred))**2 / X_test.shape[0]
    
    def save(self, directory='model.npz'):
        np.savez(directory, U=self.U, mu=self.mu, std=self.std, K=self.K)

    def load(self, directory='model.npz'):
        data = np.load(directory, allow_pickle=True)
        self.U = data["U"]
        self.mu = data["mu"]
        self.std = data["std"]
        self.K = int(data["K"])



if __name__ == "__main__":
    np.random.seed(42)
    m, n, k = 300, 10, 2
    true_Z = np.random.randn(m, k)
    true_U = np.random.randn(n, k)
    noise  = np.random.randn(m, n) * 0.5
    X      = true_Z @ true_U.T + noise
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    X[:100]   += 3
    X[100:200] -= 3
    model = PCA(K=k)
    model.fit(X)
    print(f"reconstruction error: {np.linalg.norm(X - model.reconstruction(model.transform(X)))**2 / X.shape[0]:.4f}")
    
