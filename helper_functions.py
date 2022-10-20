
import numpy as np

def relu(Z):
    """ 
    Takes an input Z and returns max(0,Z) and Z.
    Z is "cached" so that it can still be used by the backpropagation functions.
    """
    A = np.maximum(0, Z)
    assert(A.shape == Z.shape) # Necessary so that A can be used as the new Z in future layers.
    cache = Z
    return A, cache

def sigmoid(Z):
    """
    Takes an input Z and returns the sigmoid of it and a cached Z for backpropagation reasons.
    """
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache

def relu_backprop(dA, cache):
    """
    Takes an input dA which is the gradient after activation, and a cached Z.
    Returns the gradient of the cost function with respect to Z.
    """
    Z = cache
    dZ = np.array(dA, copy = True) #sets the right shape and type to dZ.    
    dZ[Z <= 0] = 0 #Need to correct for the cases wehre Z <= 0 since ReLU(.) = max(0, .)
    assert(dZ.shape == Z.shape)   
    return dZ

def sigmoid_backprop(dA, cache):
    """
    Takes an input dA which is the gradient after activation, and a cached Z.
    Returns the gradient of the cost function with respect to Z.
    """
    Z = cache
    s = 1/(1+np.exp(-Z)) #Derivative of sigmoid function.
    dZ = dA * s * (1-s)
    assert(dZ.shape == Z.shape)
    return dZ,Z
