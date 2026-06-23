import numpy as np


class GDA:
    def __init__(self):
        self.phi = None
        self.means = None
        self.covariance = None
        self.covariance_inv = None
        self.covariance_det = None
        self.classes = None
        self.n = None

    def multivariate_gaussian(self, feature, y):
        if self.means is None:
            raise Exception("Train The Model First")
        feature = np.array(feature)
        class_index = np.where(self.classes == y)[0][0]
        mean = self.means[class_index]
        return (1 / (((2 * np.pi) ** (self.n * 0.5)) * (self.covariance_det ** 0.5))) * np.exp(-0.5 * (feature - mean) @ self.covariance_inv @ (feature - mean).T)

    def fit(self, X_train, Y_train):
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if Y_train.ndim == 1:
            Y_train = Y_train.reshape(-1, 1)

        m, n = X_train.shape
        self.n = n
        Y_train = Y_train.flatten()
        self.classes = np.unique(Y_train)
        num_classes = len(self.classes)

        self.phi = np.zeros(num_classes)
        self.means = np.zeros((num_classes, n))
        M = np.zeros(X_train.shape)

        for i, cls in enumerate(self.classes):
            X_class = X_train[Y_train == cls]
            self.phi[i] = X_class.shape[0] / m
            self.means[i] = np.mean(X_class, axis=0)
            M[Y_train == cls] = self.means[i]

        self.covariance = ((X_train - M).T @ (X_train - M)) / m
        self.covariance_inv = np.linalg.inv(self.covariance)
        self.covariance_det = np.linalg.det(self.covariance)

        if num_classes == 2:
            self.mean0 = self.means[0]
            self.mean1 = self.means[1]

    def predict(self, X_test, probability=False):
        if self.means is None:
            raise Exception("Train The Model First")
        X_test = np.array(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        Y_pred = []
        Y_pred_prob = []
        for x in X_test:
            scores = []
            for i, cls in enumerate(self.classes):
                scores.append(self.multivariate_gaussian(x, cls) * self.phi[i])
            scores = np.array(scores)
            Y_pred_prob.append(scores / np.sum(scores))
            Y_pred.append(self.classes[np.argmax(scores)])

        if probability:
            return np.array(Y_pred_prob)
        return np.array(Y_pred)

    def score(self, X_test, Y_test):
        Y_pred = self.predict(X_test)
        return np.mean(Y_pred == np.asarray(Y_test).flatten())

    def save(self, directory='model.npy'):
        np.savez(directory,phi=self.phi, means=self.means, covariance=self.covariance, classes=self.classes)

    def load(self, directory='model.npy'):
        data = np.load(directory, allow_pickle=True)
        self.phi = data["phi"]
        self.means = data["means"]
        self.covariance = data["covariance"]
        self.classes = data["classes"]
        self.n = self.means.shape[1]
        self.covariance_inv = np.linalg.inv(self.covariance) # taking inverse
        self.covariance_det = np.linalg.det(self.covariance) # taking determinant of covariance

        if len(self.classes) == 2:
            self.mean0 = self.means[0]
            self.mean1 = self.means[1]
