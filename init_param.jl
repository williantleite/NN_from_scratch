function init_param(layer_dim)
  L = length(layer_dim)
  parameters = Dict()
  for l in 2:(L)
    parameters[string("W", l-1)] = rand(layer_dim[l], layer_dim[l - 1]) * 0.01
    parameters[string("b", l-1)] = zeros(layer_dim[l], 1)
    @assert size(parameters[string("W", l-1)]) == (layer_dim[l], layer_dim[l - 1]) "Weights size wrong in init_param function"
    @assert size(parameters[string("b", l-1)]) == (layer_dim[l], 1) "Bias size wrong in init_param function"
  end
  return parameters
end