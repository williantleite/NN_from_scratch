function init_param(layer_dim)
  L = length(layer_dim)
  parameters = Dict()
  for l in 1:(L - 1)
    parameters[string("W", l)] = rand(layer_dim[l], layer_dim[l + 1]) * 0.01
    parameters[string("b", l)] = zeros(layer_dim[l + 1], 1)
    @assert size(parameters[string("W", l)]) == (layer_dim[l], layer_dim[l + 1]) "Weights size wrong in init_param function"
    @assert size(parameters[string("b", l)]) == (layer_dim[l + 1], 1) "Bias size wrong in init_param function"
  end
  return parameters
end