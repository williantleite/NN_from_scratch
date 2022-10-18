import numpy as np
def cost_computation(AV,Y):
    """
    Takes AV and Y, the value of the last activation in the NN, and the true labels of the data.
    Returns the cross-entropy cost.
    """
    m=Y.shape[1]
    cost=-(1/m)*np.sum(np.dot(np.log(AV),Y.T)+np.dot(np.log(1-AV),(1-Y.T)))
    cost=np.squeeze(cost) #this makes sure that the values are not expressed as lists inside of lists.
    assert(cost.shape == ())
    return cost
