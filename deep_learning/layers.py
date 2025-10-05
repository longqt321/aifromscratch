import math
import numpy as np
from deep_learning.activation_functions import *

class Layer(object):
    
    def set_input_shape(self,shape):
        """Set the shape of the layer expects of the input in the forward pass method"""
        