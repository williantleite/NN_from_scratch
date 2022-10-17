def init_param(layer_dim):
    """
    Takes an input list with the dimensions of each layer in the network and returns a dictionary with parameters W and b.
    """
    parameters={}
    L = len(layer_dim) #how many layers there are in the network
    for l in range(1, L):
        parameters['W'+str(l)]=np.random.randn(layer_dim[l],layer_dim[l-1])*0.01
        parameters['b'+str(l)]=np.zeros((layer_dim[l],1))
        assert(parameters['W'+str(l)].shape==(layer_dim[l],layer_dim[l-1]))
        assert(parameters['b'+str(l)].shape==(layer_dim[l],1))
    return parameters
