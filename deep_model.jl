include("for_prop.jl")

function deep_model(X,parameters)
  caches = []
  A = X
  L = div(length(parameters), 2)
  for l in 1:L
    if l == L
      A_prev = A
      AV, cache = for_activation(A_prev,
                                 parameters[string("W", L)],
                                 parameters[string("b", L)],
                                 activ = "sigmoid")
      push!(caches, cache)
    else
      A_prev = A
      A, cache = for_activation(A_prev,
                                parameters[string("W", l)],
                                parameters[string("b", l)],
                                activ = "relu")
      push!(caches, cache)
    end
  end
  @assert size(AV) == (1, size(X)[1]) "AV size wrong in deep_model function"
  return AV, caches
end