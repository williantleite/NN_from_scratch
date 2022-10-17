def update(params,grads,learning_rate):
    """
    Takes params, grads, and learning rate, i.e., dictionary with parameters, dictionary with gradients
    (from deep_model_back()), and the learning rate of choice.
    Returns a dictionary with parameters W and b.
    """
    parameters=params.copy()
    L=len(parameters)//2
    for l in range(L):
        parameters['W'+str(l+1)]=parameters['W'+str(l+1)]-learning_rate*grads['dW'+str(l+1)]
        parameters['b'+str(l+1)]=parameters['b'+str(l+1)]-learning_rate*grads['db'+str(l+1)]
    return parameters
