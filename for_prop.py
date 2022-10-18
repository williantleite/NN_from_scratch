import numpy as np
from helper_functions import sigmoid,relu
def for_prop(A,W,b):
    """
    Takes inputs A, W, b, respecitively input data (previous activations), a weights matrix, and a bias vector.
    Returns Z (the element which will be used in the activation function), and cached A, W, b and 
    elements used for backpropagation.
    """
    Z=W.dot(A)+b
    assert(Z.shape==(W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return Z,cache

def for_activation(A_prev,W,b,activ):
    """
    Takes A_prev, W, b, and activ, respectively input data (or activations of the previous layer), 
    a weights matrix, a bias vector, and the activation function be used (either "sigmoid" or "relu").
    Returns the output of the activation and a cached information about the element Z and activation A.
    """
    if activ=='sigmoid':
        Z,linear_cache=for_prop(A_prev,W,b)
        A,activ_cache=sigmoid(Z)
    elif activ=='relu':
        Z,linear_cache=for_prop(A_prev,W,b)
        A,activ_cache=relu(Z)
    assert(A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(linear_cache,activ_cache)
    return A,cache
