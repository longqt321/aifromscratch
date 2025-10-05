import math
import numpy as np
from deep_learning.activation_functions import *
import copy

class Layer(object):
    
    def set_input_shape(self,shape):
        """Set the shape of the layer expects of the input in the forward pass method"""
        self.input_shape = shape
    
    def layer_name(self):
        """The name of the layers. Used in summary"""
        self.__class__.__name__

    def parameters(self):
        """The number of trainable parameters used by the layer"""
        return 0
    
    def forward_pass(self,X,traning):
        """Propogates the signal forward in the network"""
        raise NotImplemented()
    
    def backward_pass(self,accum_grad):
        """lambda"""
        raise NotImplemented()
    def output_shape(self):
        """The shape of the ouput produced by forward_pass"""
        raise NotImplemented()
    
class Dense(Layer):
    """A fully-connected NN layer.
    Parameters:
    n_units: int
        The number of neurons in the layer
    input_shape: tuple
        The expected input shape of the layer. For dense layers a single digit specifying the number
        the number of features of the input. Must be specified if it is the first layer in the network
    """

    def __init___(self,n_units,input_shape=None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None

    def initialize(self,optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit,limit,(self.input_shape[0],self.n_units))
        self.w0 = np.zeros((1,self.n_units))

        # Weight optimizers
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)
    
    def forward_pass(self,X,training=True):
        self.layer_input = X
        return X.dot(self.W) + self.w0
    
    def backward_pass(self, accum_grad):
        W = self.W

        if self.trainable:
            grad_w = self.layer_input.T.dot(accum_grad)
            grad_w0 = np.sum(accum_grad,axis=0,keepdims=True)

            self.W = self.W_opt.update(self.W,grad_w)
            self.w0 = self.w0_opt.update(self.w0,grad_w0)
        
        accum_grad = accum_grad.dot(W.T)
        return accum_grad
    
    def output_shape(self):
        return (self.n_units,)

activation_functions = {
    'relu': ReLU,
    'sigmoid': Sigmoid,
    # 'selu': SELU,
    # 'elu': ELU,
    'softmax': Softmax,
    'leaky_relu': LeakyReLU,
    'tanh': TanH,
    # 'softplus': SoftPlus
}

class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """
    def __init__(self,name):
        self.activation_name = name
        self.activation_func = activation_functions[name]()
        self.trainable = True

    def layer_name(self):
        return "Activatioon (%s)" % (self.activation_func.__class__.__name__)
    
    def forward_pass(self,X,training=True):
        self.layer_input = X
        return self.activation_func(X)
    
    def backward_pass(self, accum_grad):
        return accum_grad * self.activation_func.grad(self.layer_input)
    
    def output_shape(self):
        return self.input_shape