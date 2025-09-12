import numpy as np
import math

class l1_regularization():
    """ Regularization for Lasso Regression """
    def __init__(self,alpha):
        self.alpha = alpha
    def __call__(self,w):
        return self.alpha * np.linalg.norm(w)
    def grad(self,w):
        return self.alpha * self.sign(w)

class Regression(object):
    def __init__(self, n_iters, lr):
        self.n_iters = n_iters
        self.lr = lr
    def init_weights(self,n_features):
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit,limit,(n_features,1))
    def fit(self,X,y):
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        self.training_errors = []
        self.init_weights(n_features=X.shape[1])
        
        # Do gradient descent
        for i in range(self.n_iters):
            y_pred = X.dot(self.w)
            
            mse = np.mean(0.5 * (y-y_pred)**2 + self.regularization(self.w)) 
            self.training_errors.append(mse)

            grad_w = -np.dot(X.T,(y-y_pred))
            
            self.w -= self.lr * grad_w
        return self.training_errors
    def predict(self,X):
        X = np.concatenate((np.ones((X.shape[0],1)),X),axis=1)
        print(X.shape)
        y_pred = X.dot(self.w)
        return y_pred
class LinearRegression(Regression):
    def __init__(self,n_iters=1000,lr=0.0001):

        self.regularization = lambda x:0
        self.regularization.grad = lambda x:0

        super(LinearRegression,self).__init__(n_iters=n_iters,lr=lr)
    def fit(self,X,y):
        super(LinearRegression,self).fit(X,y)