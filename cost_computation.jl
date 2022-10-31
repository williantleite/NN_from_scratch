function cost_computation(AV,Y)
  m = length(Y)
  cost = -(1/m) * sum(log.(AV) * Y' .+ log.((1 .- AV)) * (1 .- Y)')
  return cost
end