class GradientAscent:
    def __init__(self, alpha=1e-4, weights=1):
        self.alpha = alpha
        self.weights = weights

    def batch_gradient_descent(self, X_train, Y_train, y_pred):
        return self.alpha * (X_train.T @ (self.weights * (y_pred - Y_train)))