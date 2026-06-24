import numpy as np

class FactorAnalysis:
    def __init__(self, d, epoch=100):
        self.X = None # (m,n)
        self.d = d # no of latent factors # 
        self.psi = None ## psi is a diagonal matrix here Ai!=j (n,n)
        self.mu = None #(1,n)
        self.lamda = None #(n,d)
        self.epoch = epoch 

    def fit(self, X_train, epochs=None, verbose=True):
        X_train = np.array(X_train)
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)
        self.X = X_train # (m,n)
        self.psi =  np.diag(np.var(X_train, axis=0)) ## psi is a diagonal matrix here Ai!=j (n,n)
        self.mu = np.mean(X_train, axis=0) #(1,n)
        self.lamda = np.random.randn(self.X.shape[1],self.d) * 0.01 #(n,d)
        if epochs is not None:
            self.epoch = epochs
        self.optimizer(verbose=verbose)

    def optimizer(self, verbose=True):
        if self.X is None:
            raise Exception("Train The Model First")
        old_loss = -np.inf
        for epoch in range(self.epoch):
            # E-STEP
            # (Z | X) ~ (0,u) , sigma)
            mu = np.zeros((self.X.shape[0], self.d)) # (m,d)
            some_var = np.linalg.inv(((self.lamda @ self.lamda.T) + self.psi)) #(n,n)
            mu =  (self.X - self.mu) @ some_var @ self.lamda  # (m,n) @  (d,n)(n,n) (d,n) => (m,n)(n,d) => (m,d)
            sigma = np.eye(self.d) - self.lamda.T @ some_var @ self.lamda

            # M STEP
            self.mu = np.sum(self.X, axis=0) / self.X.shape[0]
            self.lamda = (self.X.T @ mu) @ np.linalg.inv(np.sum((sigma + mu[:, :, None] * mu[:, None , :]),axis=0))
            psi_diag = np.diag((self.X.T @ self.X - self.lamda @ mu.T @ self.X )/self.X.shape[0])
            self.psi = np.diag(psi_diag)
            self.psi += 1e-8 * np.eye(self.X.shape[1])
            new_loss = self.log_likelyhood()
            if np.abs(new_loss - old_loss) < 1e-4:
                if verbose:
                    print(f"[CONVERGED] Epoch {epoch} | Loss : {new_loss:.4f}")
                break
            old_loss = new_loss
            if verbose:
                print(f"Epoch {epoch:4d} | Loss: {new_loss:.4f}")
        
    def multivariate_gaussian(self,x, mu, sigma): # P(X) ~ N(xi; u,A.AT + psi)
        n = x.shape[1]
        diff = x - mu                                                          # (m,n)
        mahal = np.sum(diff @ np.linalg.inv(sigma) * diff, axis=1)            # (m,)
        coef = 1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5)
        return coef * np.exp(-0.5 * mahal)
        
        # just implement # p(x) = (1 / ((2π)^(n/2) * |Σ|^(1/2))) * exp(-1/2 * (x-μ)^T Σ^(-1) (x-μ))
            
    
    def log_likelyhood(self):
        if self.X is None:
            raise Exception("Train The Model First")
        return np.sum(np.log(self.multivariate_gaussian(self.X, self.mu , (self.lamda @ self.lamda.T) + self.psi)))
    
    
    def transform(self, x):
        if self.X is None:
            raise Exception("Train The Model First")
        x = np.array(x)
        if (x.ndim == 1):
            x = x.reshape(1, -1)
        return (x - self.mu) @ np.linalg.inv((self.lamda @ self.lamda.T) + self.psi) @ self.lamda

    def reconstruction_error(self, x):
        if self.X is None:
            raise Exception("Train The Model First")
        x = np.array(x)
        if (x.ndim == 1):
            x = x.reshape(1, -1)
        x_hat = (self.transform(x) @ self.lamda.T) + self.mu 
        return np.linalg.norm(x - x_hat) ** 2 / x.shape[0]
    
    def save(self, directory='model.npz'):
        np.savez(directory, d=self.d, psi=self.psi, mu=self.mu, lamda=self.lamda)

    def load(self, directory='model.npz'):
        data = np.load(directory, allow_pickle=True)
        self.d = int(data["d"])
        self.psi = data["psi"]
        self.mu = data["mu"]
        self.lamda = data["lamda"]
        self.X = np.zeros((1, self.mu.shape[0]))



if __name__ == "__main__":
    np.random.seed(42)
    m, n, d = 300, 10, 3
    true_Z = np.random.randn(m, d)
    true_L = np.random.randn(n, d)
    noise  = np.random.randn(m, n) * 0.5
    X      = true_Z @ true_L.T + noise
    labels = np.array([0]*100 + [1]*100 + [2]*100)
    X[:100]  += 3
    X[100:200] -= 3
    model = FactorAnalysis(d=d, epoch=200)
    model.fit(X)
    print(f"log likelihood:       {model.log_likelyhood():.4f}")
    print(f"reconstruction error: {model.reconstruction_error(X):.4f}")
    
