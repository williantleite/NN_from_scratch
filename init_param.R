fill.matrix<-function(expr,nrow=1,ncol=1){
  matrix(eval(expr,envir=list(x=nrow*ncol)),nrow=nrow,ncol=ncol)
}
init_param<-function(layer_dim){
  L<-length(layer_dim)
  parameters=vector("list",0)
  for (l in 1:(L-1)){
    n=layer_dim[l]*layer_dim[l+1]
    parameters[[paste0("W",l)]]<-fill.matrix(rnorm(n=n)*0.01,nrow=layer_dim[l+1],ncol=layer_dim[l])
    parameters[[paste0("b",l)]]<-matrix(0,layer_dim[l+1],1)
  }
  return(parameters)
}
