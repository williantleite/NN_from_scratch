source("helper_functions.R")
for_prop<-function(A,W,b){
    Z<-sweep((W%*%A),MARGIN=1,STATS=b,FUN="+")
    cache<-list(A,W,b)
    return(list(Z,cache))
}

for_activation<-function(A_prev,W,b,activ){
    if(activ=='sigmoid'){
        calc<-for_prop(A_prev,W,b)
        Z<-calc[[1]]
        linear_cache<-calc[[2]]
        calc2<-sigmoid(Z)
        A<-calc2[[1]]
        activ_cache<-calc2[[2]]
    } else if (activ=='relu'){
        calc<-for_prop(A_prev,W,b)
        Z<-calc[[1]]
        linear_cache<-calc[[2]]
        calc2<-relu(Z)
        A<-calc2[[1]]
        activ_cache<-calc2[[2]]
    }
    cache=list(linear_cache,activ_cache)
    return(list(A,cache))
}
