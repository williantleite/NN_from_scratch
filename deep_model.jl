include("for_prop.jl")

function deep_model(X, parameters)
  caches = []
  A = X
  L = div(length(parameters), 2)
  for l in 1:(L-1)
    A_prev = A
    A, cache = for_activation(A_prev,
                              parameters[string("W", l)],
                              parameters[string("b", l)],
                              "relu")
    push!(caches, cache)
  end
  AV, cache = for_activation(A,
                             parameters[string("W", L)],
                             parameters[string("b", L)],
                             "sigmoid")
  push!(caches, cache)
  @assert size(AV) == (1, size(X)[2]) "AV size wrong in deep_model function"
  return AV, caches
end