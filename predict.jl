include("deep_model.jl")
function predict(X, y, parameters)
  m = size(X)[2]
  n = div(length(parameters), 2)
  p = Int[]
  probs, caches = deep_model(X, parameters)
  for i in 1:size(probs)[2]
    if probs[i] > 0.25
      push!(p, 1)
    else
      push!(p, 0)
    end
  end
  print(string("Accuracy ", sum((p .== y)/m)))
  return p
end