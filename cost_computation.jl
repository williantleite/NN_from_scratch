function cost_computation(AV, Y)
  Y = Y'
  m = size(Y)[2]
  cost = -(1/m) * sum(log.(AV) * Y' .+ log.((1 .- AV)) * (1 .- Y'))
  return cost
end