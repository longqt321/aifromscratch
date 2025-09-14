import numpy as np

class Sigmoid():
    def __call__(self,x):
        return 1 / (1 + np.exp(-x))
    def grad(self,x):
        return self.__call__(x)*(1 - self.__call__(x))
    
class Softmax():
    def __call__(self,x):
        e_x = np.exp(x - np.max(x,axis=-1,keepdims=True))
        return e_x / np.sum(e_x,axis=-1,keepdims=True)
    def grad(self,x):
        p = self.__call__(x)
        return p*(1-p)

class TanH():
    def __call__(self,x):
        pass
    def grad(self,x):
        pass

class ReLU():
    def __call__(self,x):
        return np.where(x>=0,x,0)
    def grad(self,x):
        return np.where(x>=0,1,0)
    
class LeakyReLU():
    def __init__(self,alpha=0.2):
        self.alpha = alpha
    def __call__(self, x):
        return np.where(x>=0,x,self.alpha*x)
    def grad(self,x):
        return np.where(x>=0,1,self.alpha)