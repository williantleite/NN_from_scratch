cost_computation<-function(AV,Y){
    m<-dim(Y)[2]
    cost=-(1/m)*sum(log(AV)%*%t(Y)+log((1-AV))%*%(1-t(Y)))
    cost<-unlist(cost)
    return(cost)
}
