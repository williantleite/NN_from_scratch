source("helper_functions.R")
for_prop<-function(A,W,b){
    Z<-(W%*%A)+b
    cache<-list(A,W,b)
    return(list(Z,cache))
}

for_activation<-function(A_prev,W,b,activ){
    if(activ=='sigmoid'){
        Z<-for_prop(A_prev,W,b)[[1]]
        linear_cache<-for_prop(A_prev,W,b)[[2]]
        calc<-sigmoid(Z)
        A<-calc[[1]]
        activ_cache<-calc[[2]]
    } else if (activ=='relu'){
        Z<-for_prop(A_prev,W,b,activ)[[1]]
        linear_cache<-for_prop(A_prev,W,b)[[2]]
        calc<-relu(Z)
        A<-calc[[1]]
        activ_cache<-calc[[2]]
    }
    cache=list(linear_cache,activ_cache)
    return(list(A,cache))
}
