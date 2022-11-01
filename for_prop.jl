include("helper_functions.jl")
using LinearAlgebra

function for_prop(A, W, b)
  Z = dot(W, A) + b
  @assert size(Z) == (size(W)[1], size(A)[2])
  cache = (A, W, b)
  return Z, cache
end

function for_activation(A_prev, W, b, activ)
  if activ == "sigmoid"
    Z, linear_cache = for_prop(A_prev, W, b)
    A, activ_cache = sigmoid(Z)
  elseif activ == "relu"
    Z, linear_cache = for_prop(A_prev, W, b)
    A, activ_cache = relu(Z)
  end
  @assert size(A) == (size(W)[1], size(A_prev)[2]) "Activation size wrong in for_activation function"
  cache = (linear_cache, activ_cache)
  return A, cache
end