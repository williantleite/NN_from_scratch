#one could potentially use the `sigmoid` library for the relu and sigmoid functions, but they are very simple to implement too.
relu<-function(Z){
  A<-pmax(Z,0)
  cache<-Z
  return(list(A,cache))
}
sigmoid<-function(Z){
  A<-1/(1+exp(Z))
  cache<-Z
  return(list(A,cache))
}
relu_backprop<-function(dA, cache){
  Z=cache
  dZ<-dA
  dZ[Z<=0]<-0
  return(dZ)
}
sigmoid_backprop<-function(dA,cache){
  Z<-cache
  s<-1/(1+exp(-Z))
  dZ<-dA*s*(1-s)
  return(dZ)
}
