function update(params, grads, learning_rate)
  parameters = copy(params)
  L = div(length(parameters), 2)
  for l in 1:L
    parameters[string("W", l)] = parameters[string("W", l)] - learning_rate * grads[string("dW", l)]'
    parameters[string("b", l)] = parameters[string("b", l)] - learning_rate * grads[string("db", l)]
  end
  return parameters
end