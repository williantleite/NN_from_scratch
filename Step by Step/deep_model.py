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
            A,cache=for_activation(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
            caches.append(cache)
        AV,cache=for_activation(A,parameters['W'+str(l)],parameters['b'+str(l)],activation='sigmoid')
        caches.append(cache)
        return AV,caches
