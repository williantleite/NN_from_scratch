function relu(Z)
  A = max(Z, 0)
  cache = Z
  @assert size(A) == size(Z) "Sizes don't match in relu function"
  return A, cache
end

function sigmoid(Z)
  A = 1/(1 + exp.(Z))
  cache = Z
  return A, cache
end

function relu_backprop(dA, cache)
  Z = cache
  dZ = dA
  dZ[Z .<= 0] .= 0
  @assert size(dZ) == size(Z) "Sizes don't match in relu_backprop function"
  return dZ
end

function sigmoid_backprop(dA, cache)
  Z = cache
  s = 1/(1 + exp.(-Z))
  dZ = dA * s * (1-s)
  @assert size(dZ) == size(Z) "Sizes don't match in sigmoid_backprop function"
  return dZ
end