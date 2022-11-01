include("update.jl")
include("back_prop.jl")
include("deep_model.jl")
include("init_param.jl")
include("cost_computation.jl")
function dense_nn(X, Y, layers_dims, learning_rate, num_iterations, print_cost)
  costs = Float64[]
  parameters = init_param(layers_dims)
  for i in 1:num_iterations
    AV, caches = deep_model(X, parameters)
    cost = cost_computation(AV, Y)
    grads = deep_model_back(AV, Y, caches)
    parameters = update(parameters, grads, learning_rate)
    if print_cost && i%100==0 || i==num_iterations-1
      print(string("Cost after iteration ",i,": ",cost))
    elseif i%100==0 || i==num_iterations
      push!(costs, cost)
    end
  end
  return parameters, costs
end