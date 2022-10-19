fill.matrix<-function(expr,nrow=1,ncol=1){
    matrix(eval(expr,envir=list(x=nrow*ncol)),nrow=nrow,ncol=ncol)
}
init_param<-function(layer_dim){
    parameters<-list()
    L<-length(layer_dim)
    for (l in 1:L){
        parameters[paste("W",l,sep="")]<-fill.matrix(rnorm(n=layer_dim[l]*layer_dim[l-1],mean=0)*0.01,layer_dim[l],layer_dim[l-1])
        parameters[paste("b",l,sep="")]<-matrix(0,layer_dim[l],1)
    }
    return(parameters)
}
