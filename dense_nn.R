source("update.R")
source("back_prop.R")
source("deep_model.R")
source("init_param.R")
source("cost_computation.R")

dense_nn<-function(X,Y,layers_dims,learning_rate=0.0075,num_iterations=5000,print_cost=False){
    costs<-list()
    parameters<-init_param(layers_dims)
    for(i in 1:num_interations){
        calc<-deep_model(X,parameters)
        AV<-calc[[1]]
        caches<-calc[[2]]
        cost<-cost_computation(AV,Y)
        grads<-deep_model_back(AV,Y,caches)
        parameters<-update(parameters,grads,learning_rate)
        if(print_cost && i%%100==0 | i==num_interations-1){
            print("Cost after iteration ",i,": ",cost)
        }
        if(i%%100==0 | i==num_iterations){
            append(costs,cost)
        }
    }
    return(list(parameters,costs))
}
