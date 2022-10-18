import numpy as np
from helper_functions import relu_backprop, sigmoid_backprop
def back_prop(dZ,cache):
    """
    Takes dZ and cache, the gradient of the cost and the cached values to produce the Z element.
    Returns dA_prev, dW, and db, respectively the gradient with respect to the activation, weights, and biases.
    """
    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=(1/m)*np.dot(dZ,A_prev.T)
    db=(1/m)*np.sum(dZ,axis=1,keepdims=True)
    dA_prev=np.dot(W.T,dZ)
    assert(dA_prev.shape==A_prev.shape)
    assert(dW.shape==W.shape)
    assert(db.shape==b.shape)
    return dA_prev,dW,db

def back_activ(dA,cache,activ):
    """
    Takes dA (gradient of the activation element), cache (saved values for activation and forward prop),
    and the activation type (sigmoid or relu).
    Returns dA_prev, dW, and db. Respectively the gradient with respect to the actgivation, weights and biases.
    """
    for_cache,activ_cache=cache
    if activ=='relu':
        dZ=relu_backprop(dA,activ_cache)
        dA_prev,dW,db=back_prop(dZ,for_cache)
    elif activ=='sigmoid':
        dZ=sigmoid_backprop(dA,activ_cache)
        dA_prev,dW,db=back_prop(dZ,for_cache)
    return dA_prev,dW,db

def deep_model_back(AV,Y,caches):
    """
    Takes AV, Y, and caches. The activation value of the output layer (produced in deep_model()), a vector with the true
    labels, and the caches containing all the for_prop() information for each activation function.
    Returns a dictionary with gradients dA, dW, and db.
    """
    grads={}
    L=len(caches)
    m=AV.shape[1]
    Y=Y.reshape(AV.shape)
    dAL=-(np.divide(Y,AV)-np.divide(1-Y,1-AV)) #the derivative of the cost function I talked about in the preamble of this section
    present_cache=caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=back_activ(dAL,present_cache,activ="sigmoid")
    for l in reversed(range(L-1)):
        present_cache=caches[l]
        dA_prev_temp,dW_temp,db_temp=back_activ(grads['dA'+str(l+1)],present_cache,activ='relu')
        grads['dA'+str(l)]=dA_prev_temp
        grads['dW'+str(l+1)]=dW_temp
        grads['db'+str(l+1)]=db_temp
    return grads
