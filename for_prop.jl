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
    calc = for_prop(A_prev, W, b)
    Z = calc[1]
    linear_cache = calc[2]
    calc2 = sigmoid(Z)
    A = calc2[1]
    activ_cache = calc2[2]
  elseif activ == "relu"
    calc = for_prop(A_prev, W, b)
    Z = calc[1]
    linear_cache = calc[2]
    calc2 = relu(Z)
    A = calc2[1]
    activ_cache = calc2[2]
  end
  @assert size(A) == (size(W)[1], size(A_prev)[2]) "Activation size wrong in for_activation function"
  cache = (linear_cache, activ_cache)
  return A, cache
end