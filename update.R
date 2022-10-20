update<-function(params,grads,learning_rate){
    parameters<-params
    L<-length(parameters)%/%2
    for (l in 1:L){
        parameters[[paste0("W",l+1)]]<-parameters[[paste0("W",l+1)]]-learning_rate*grads[[paste0('dW',l+1)]]
        parameters[[paste0("b",l+1)]]<-parameters[[paste0("b",l+1)]]-learning_rate*grads[[paste0('db',l+1)]]
    }
    return(parameters)
}
