import numpy as np

try:
    from neutronix.unsupervised.kmeans import KMeans
except ImportError:
    try:
        from unsupervised.kmeans import KMeans
    except ImportError:
        from kmeans import KMeans
class GMM:
    def __init__(self, cluster, epoch=1000):
        self.X = None
        self.j = cluster # total of cluster
        self.kmeans = None
        self.epoch = epoch
        ## parameter initalization for best results using kmeans algorithm
        self.mu = None
        self.phi = None
        self.sigma = None

    def initialize_parameters(self, X_train):
        self.X = X_train
        self.kmeans = KMeans(self.j)
        indices = np.random.choice(X_train.shape[0], size=self.j, replace=False)
        self.kmeans.centroids = X_train[indices]
        self.kmeans.optimizer_func(X_train, verbose=False, epochs=10)
        self.mu = self.kmeans.centroids.copy() 
        self.phi = np.zeros(self.j)
        # # len(C[ C == j]) / X.shape[0]
        # C = kmeans.C
        # for i in range(self.j):
        #     self.phi[i] = len(C[C==i]) / kmeans.X.shape[0]

        self.phi = np.bincount(self.kmeans.C, minlength=self.j) / len(self.kmeans.C)
        # j=1 
        self.sigma = np.zeros((self.j , self.X.shape[1], self.X.shape[1]))
        for j in range(self.j):
            if np.count_nonzero(self.kmeans.C == j) == 0:
                self.sigma[j] = np.eye(self.X.shape[1])
            else:
                self.sigma[j] = ((self.X[self.kmeans.C ==j]-self.mu[j]).T @(self.X[self.kmeans.C ==j]-self.mu[j])) /np.count_nonzero(self.kmeans.C == j)
        self.sigma += 1e-8 * np.eye(self.X.shape[1]) # epsilon stablization

    def multivariate_gaussian(self,x, sigma , mu): # P(xi | zi = j) ~ N(u,E)
        if self.X is None:
            raise Exception("Train The Model First")
        return  (1/(((2*np.pi)**(self.X.shape[1]*0.5))*(np.linalg.det(sigma)**0.5))) * np.exp(-0.5 * np.inner((x-mu), np.linalg.inv(sigma) @ (x-mu)))
        # just implement # p(x) = (1 / ((2π)^(n/2) * |Σ|^(1/2))) * exp(-1/2 * (x-μ)^T Σ^(-1) (x-μ))
    
    def fit(self, X_train, epochs=None, verbose=True):
        X_train = np.array(X_train)
        if (X_train.ndim == 1):
            X_train = X_train.reshape(-1, 1)

        if (X_train.shape[0] < self.j):
            raise Exception("[ERROR] Number of Clusters Greater Than The Sample Space")

        if epochs is not None:
            self.epoch = epochs

        self.initialize_parameters(X_train)
        self.train(verbose=verbose)

    def train(self, verbose=True):
        if self.X is None:
            raise Exception("Train The Model First")
        W = np.zeros((self.X.shape[0], self.j))
        old_loss = -np.inf
        for epoch in range(self.epoch):
            #  E-Step
            for j in range(self.j):
                for i in range(self.X.shape[0]):
                    numerator = self.multivariate_gaussian(self.X[i], self.sigma[j], self.mu[j]) * self.phi[j]
                    denominator = 0
                    for l in range(self.j):
                        denominator+=self.multivariate_gaussian(self.X[i], self.sigma[l], self.mu[l]) * self.phi[l]
                    W[i,j] = numerator/denominator
            # M-Step
            
            for j in range(self.j): # updating phi
                sum_phi = 0
                sum_mu = np.zeros(self.mu.shape[1])
                sum_sigma = np.zeros((self.X.shape[1],self.X.shape[1]))
                for i in range(self.X.shape[0]):
                    sum_phi+=W[i,j]
                    sum_mu += W[i,j] * self.X[i]
                    sum_sigma += W[i,j] * np.outer((self.X[i] - self.mu[j]) ,(self.X[i] - self.mu[j]))
                self.phi[j] = sum_phi/self.X.shape[0]
                if sum_phi > 0:
                    self.mu[j] = sum_mu/sum_phi
                    self.sigma[j] = sum_sigma/sum_phi
                self.sigma[j] += 1e-6 * np.eye(self.X.shape[1])
        
            new_loss = self.log_likelyhood()
            if np.abs(new_loss -old_loss) < 1e-4:
                if verbose:
                    print(f"[CONVERGED] Epoch {epoch} | Loss : {new_loss:.4f}")
                break
            old_loss = new_loss
            if(verbose):
                print(f"Epoch {epoch:4d} | Loss: {self.log_likelyhood():.4f}")
    
    def log_likelyhood(self):
        likelyhood = 0
        for i in range(self.X.shape[0]):
            temp =0
            for j in range(self.j):
               temp+= self.phi[j] * self.multivariate_gaussian(self.X[i], self.sigma[j], self.mu[j]) 
            likelyhood += np.log(temp)
        return likelyhood
    
    def predict(self, X_test, probability=False):
        if self.X is None:
            raise Exception("Train The Model First")
        X_test = np.array(X_test)
        if (X_test.ndim == 1):
            X_test = X_test.reshape(1, -1)

        pred = []
        probs = []
        for x in X_test:
            W = np.zeros((self.j)) # [1,2,3]
            for j in range(self.j):
                numerator = self.multivariate_gaussian(x, self.sigma[j], self.mu[j]) * self.phi[j]
                denominator = 1e-8
                for l in range(self.j):
                    denominator+=self.multivariate_gaussian(x, self.sigma[l], self.mu[l]) * self.phi[l]
                W[j] = numerator/denominator
            probs.append(W)
            pred.append(np.argmax(W))

        if probability:
            return np.array(probs)
        return np.array(pred)
    
    def save(self, directory='model.npz'):
        np.savez(directory, mu=self.mu, phi=self.phi, sigma=self.sigma, j=self.j)
    
    def load(self, directory='model.npz'):
        data = np.load(directory, allow_pickle=True)
        self.mu = data["mu"]
        self.phi = data["phi"]
        self.sigma = data["sigma"]
        self.j = int(data["j"])
        self.X = np.zeros((1, self.mu.shape[1]))



if __name__ == "__main__":
    gmm = GMM(cluster=2)
    X = np.array([
        [2.1, 1.9], [1.8, 2.2], [2.0, 2.0], [2.3, 1.7], [1.9, 2.1],
        [2.2, 2.3], [1.7, 1.8], [2.4, 2.1], [1.8, 1.9], [2.1, 2.2],
        # cluster 2 around 8,8
        [8.1, 7.9], [7.8, 8.2], [8.0, 8.0], [8.3, 7.7], [7.9, 8.1],
        [8.2, 8.3], [7.7, 7.8], [8.4, 8.1], [7.8, 7.9], [8.1, 8.2],
    ])
    gmm.fit(X)
    example = eval(input("enter your example pair (x1,x2)"))
    probs = gmm.predict(example, probability=True).flatten()
    for i, p in enumerate(probs):
        print(f"Cluster {i}: {p:.3f}")
