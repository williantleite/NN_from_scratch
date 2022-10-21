source("deep_model.R")
predict<-function(X,y,parameters){
    m<-dim(X)[2]
    n<-length(parameters)%/%2
    p<-matrix(0,1,m)
    dm<-deep_model(X,parameters)
    probs<-dm[[1]]
    caches<-dm[[2]]
    for (i in 1:dim(probs)[2]){
        if(probs[1,i]>0.5){
            p[1,i]=1
        } else {
            p[1,i]=0
        }
    }
    print(paste("Accuracy ", sum((p==y)/m)))
    return(p)
}
