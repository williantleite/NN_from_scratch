include("helper_functions.jl")
using LinearAlgebra

function back_prop(dZ, cache)
  A_prev, W, b = cache
  m = size(A_prev)[2]
  dW = (1/m) * dot(dZ, A_prev')
  db = (1/m) * sum(dZ)
  dA_prev = dot(W', dZ)
  @assert size(dA_prev) == size(A_prev) "dA_prev size wrong in back_prop function"
  @assert size(dW) == size(W) "dW size wrong in back_prop function"
  @assert size(db) == size(b) "db size wrong in back_prop function"
  return dA_prev, dW, db
end

function back_activ(dA, cache, activ)
  for_cache, activ_cache = cache
  if activ == "relu"
    dZ = relu_backprop(dA, activ_cache)
    dA_prev, dW, db = back_prop(dZ, for_cache)
  elseif activ == "sigmoid"
    dZ = sigmoid_backprop(dA, activ_cache)
    dA_prev, dW, db = back_prop(dZ, for_cache)
  end
  return dA_prev, dW, db
end

function deep_model_back(AV, Y, caches)
  grads = Dict()
  L = length(caches)
  m = size(AV)[2]
  Y = reshape(Y, 1, length(AV))
  dAL = -(Y ./ AV) - ((1 .- Y) ./ (1 .- AV))
  present_cache = caches[L]
  grads[string("dA", L)], grads[string("dW", L)], grads[string("db", L)] = back_activ(dAL, present_cache, activ = "sigmoid")
  for l in (L - 1):-1:1
    present_cache = caches[l]
    dA_prev_temp, dW_temp, db_temp <-back_activ(grads[string("dA", l+1)], present_cache, activ = "relu")
    grads[string("dA", l)] = dA_prev_temp
    grads[string("dW", l)] = dW_temp
    grads[string("db", l)] = db_temp
  end
  return grads
end