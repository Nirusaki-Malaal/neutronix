import numpy as np

# NAIVE BAYES IS IMPLEMENTED WITH A BUILTIN OPTIMIZER FUNCTION

class KMeans:
    def __init__(self, K):
        self.K = K
        self.centroids = None
        self.optimizer = self.optimizer_func

    def loss(self,X_train, c, u):
        loss = np.sum(np.linalg.norm(X_train - u[c], axis=1)**2)
        return loss
    
    def optimizer_func(self, X_train , verbose=True, epochs=1000):
        for epoch in range(epochs):

            old_centroids = self.centroids.copy()
            
            C = np.zeros(X_train.shape[0], dtype=int)
            distances = np.linalg.norm(X_train[:, np.newaxis] - self.centroids,axis=2) 
            C = np.argmin(distances, axis=1) 

            self.C = C

            for j in range(self.K):
                numerator = np.zeros(X_train.shape[1])
                denominator = 0
                for i in range(X_train.shape[0]):
                    numerator+= int(C[i] == j) * X_train[i]
                    denominator+= int(C[i] == j)
                if denominator > 0:
                    self.centroids[j] = numerator / denominator
                else:
                    idx = np.random.choice(X_train.shape[0])
                    self.centroids[j] = X_train[idx]

            if np.allclose(old_centroids, self.centroids):
                if verbose:
                    print(f"[CONVERGED] Epoch {epoch} | Loss : {self.loss(X_train, C, self.centroids):.4f}")
                break
            
            if verbose:
                print(f"Epoch {epoch:4d} | Loss: {self.loss(X_train, C, self.centroids):.4f}")
    
    
    def fit(self,X_train, epochs=1000):
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)
        
        if (X_train.shape[0] < self.K):
            raise Exception("[ERROR] Number of Clusters Greater Than The Sample Space")

        m,n = X_train.shape
        
        if self.centroids is None:
            indices = np.random.choice(X_train.shape[0], size=self.K, replace=False)
            self.centroids = X_train[indices]
        if self.optimizer != self.optimizer_func:
            raise Exception("[ERROR] KMEANS CURRENTLY ONLY SUPPORTS THE BUILT IN OPTIMIZER")
        self.optimizer(X_train , epochs=epochs)

    def predict(self, X_test):
        X_test = np.array(X_test)
        if (X_test.ndim == 1):
            X_test = X_test.reshape(-1, 1)
            return np.argmin((np.linalg.norm(self.centroids - X_test, axis=1))**2)
        results = []
        for i in range(X_test.shape[0]):
            label = np.argmin((np.linalg.norm(self.centroids - X_test[i], axis=1))**2)
            results.append(label)
        return np.array(results)
    
    def save(self, directory='model.npz'):
        np.savez(directory, centroids=self.centroids)

    def load(self, directory='model.npz'):
        data = np.load(directory, allow_pickle=True)
        self.centroids = data["centroids"]