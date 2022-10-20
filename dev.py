import numpy as np
import ipdb

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
    return dZ

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

def deep_model(X,parameters):
    """
    Takes X and parameters, respectively the data and the initialized parameters from init_param() function.
    Returns AV and caches, the activation value at the end of the architecture, and the cached values of every layer.
    """
    caches=[]
    A=X
    L=len(parameters)//2 #The floor division gives the number of layers in the network
    for l in range(1,L):
        A_prev=A
        A,cache=for_activation(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activ='relu')
        caches.append(cache)
    AV,cache=for_activation(A,parameters['W'+str(L)],parameters['b'+str(L)],activ='sigmoid')
    caches.append(cache)
    assert(AV.shape==(1,X.shape[1]))
    return AV,caches

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

def back_prop(dZ,cache):
    """
    Takes dZ and cache, the gradient of the cost and the cached values to produce the Z element.
    Returns dA_prev, dW, and db, respectively the gradient with respect to the activation, weights, and biases.
    """
    ipdb.set_trace()
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

import os
import cv2
def create_dataset(img_folder,IMG_WIDTH=32,IMG_HEIGHT=32):
    """
    Takes the path to the folder containing all images.
    Returns a list of images and a list of classes.
    """
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder,dir1)):
            image_path= os.path.join(img_folder,dir1,file)
            image= cv2.imread(image_path,cv2.COLOR_BGR2RGB)#For grayscale use cv2.IMREAD_GRAYSCALE
            if image is None:
                continue
            if len(image.shape) ==2: #This effectively removes the grayscale images since they do not have a third dimension
                continue
            image=cv2.resize(image,(IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_AREA)
            image=np.array(image)
            image=image.astype('float32')
            image/=255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name

IMG_WIDTH=32
IMG_HEIGHT=32
img_folder=r"C:\\Users\\wtrindad\\source\\repos\\NN_from_scratch\\PetLite"
img_data,class_name=create_dataset(img_folder,IMG_WIDTH,IMG_HEIGHT)

len(img_data)
img_data = np.asarray(img_data)
img_data.shape

target_dict={'Cat':1,'Dog':0} #Let's encode our labels
class_name=[target_dict[class_name[i]] for i in range(len(class_name))]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(img_data,class_name,test_size=.15)

x_train_flat = x_train.reshape(x_train.shape[0], -1).T
print(x_train_flat.shape)

x_test_flat = x_test.reshape(x_test.shape[0], -1).T
print(x_test_flat.shape)

y_train = np.asarray(y_train).reshape(1,-1)
print(y_train.shape)

y_test = np.asarray(y_test).reshape(1,-1)
print(y_test.shape)

layer_dims = [x_train_flat.shape[0],20,7,5,1]

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

parameters,costs=dense_nn(x_train_flat,y_train,layer_dims,learning_rate=0.3,num_iterations=1,print_cost=True)
