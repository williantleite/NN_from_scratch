source("for_prop.R")
deep_model<-function(X,parameters){
  caches<-list()
  A<-X
  L<-length(parameters)%/%2
  for (l in 1:(L)){
    if (l==L){
      A_prev<-A
      calc<-for_activation(A_prev,parameters[[paste0("W",L)]],parameters[[paste0("b",L)]],activ='sigmoid')
      AV<-calc[[1]]
      cache<-calc[[2]]
      caches[l]<-list(cache)
    } else {
    A_prev<-A
    calc<-for_activation(A_prev,parameters[[paste0("W",l)]],parameters[[paste0("b",l)]],activ='relu')
    A<-calc[[1]]
    cache<-calc[[2]]
    caches[l]<-list(cache)
    }
  }
  return(list(AV,caches))
}
