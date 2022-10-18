import numpy as np
from deep_model import deep_model
def predict(X,y,parameters):
    """
    Takes X (the data), y (the labels), and parameters (from the dense_nn() function).
    Returns predictions p for the chosen data set X.
    """
    m=X.shape[1]
    n=len(parameters)//2
    p=np.zeros((1,m))
    probs,caches=deep_model(X, parameters)
    for i in range(0,probs.shape[1]):
        if probs[0,i]>0.5:
            p[0,i]=1
        else:
            p[0,i]=0
    print("Accuracy: "  + str(np.sum((p == y)/m)))   
    return p
