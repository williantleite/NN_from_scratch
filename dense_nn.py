import numpy as np
from update import update
from deep_model import deep_model
from init_param import init_param
from back_prop import deep_model_back
from cost_computation import cost_computation

def dense_nn(X,Y,layers_dims,learning_rate=0.0075,num_iterations=5000,print_cost=False):
    """
    Takes X (the data), Y (the labels), a list with the architecture, a given learning rate, number of iterations, and you get to choose whether to print the cost every 100 steps or not.
    Returns the parameters of the model which can then be used to make predictions.
    """
    costs=[]
    parameters=init_param(layers_dims)
    for i in range(0,num_iterations):
        AV,caches=deep_model(X,parameters)
        cost=cost_computation(AV,Y)
        grads=deep_model_back(AV,Y,caches)
        parameters=update(parameters,grads,learning_rate)
        if print_cost and i%100==0 or i==num_iterations-1:
            print("Cost after iteration {}: {}".format(i,np.squeeze(cost)))
        if i%100==0 or i==num_iterations:
            costs.append(cost)
    return parameters,costs
