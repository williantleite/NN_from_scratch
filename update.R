update<-function(params,grads,learning_rate){
    parameters<-params
    L<-length(parameters)%/%2
    for (l in 1:L){
        parameters[paste("W",l+1,sep="")]<-parameters[paste("W",l+1,sep="")]-learning_rate*grads[paste('dW',l+1,sep="")]
        parameters[paste("b",l+1,sep="")]<-parameters[paste("b",l+1,sep="")]-learning_rate*grads[paste('db',l+1,sep="")]
    }
    return(parameters)
}
