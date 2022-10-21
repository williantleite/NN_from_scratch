update<-function(params,grads,learning_rate){
  parameters<-params
  L<-length(parameters)%/%2
  for (l in 1:L){
    parameters[[paste0("W",l)]]<-parameters[[paste0("W",l)]]-learning_rate*grads[[paste0('dW',l)]]
    parameters[[paste0("b",l)]]<-parameters[[paste0("b",l)]]-learning_rate*grads[[paste0('db',l)]]
  }
  return(parameters)
}
