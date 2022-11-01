include("deep_model.jl")
function predict(X, y, parameters)
    m  = size(X)[2]
    n = div(length(parameters), 2)
    p = zeros(1,m)
    probs, caches = deep_model(X, parameters)
    for i in 1:dim(probs)[2]
        if probs[1,i] > 0.5
            p[1, i] = 1
        else
            p[1, i] = 0
        end
    end
    print(string("Accuracy ", sum((p==y)/m)))
    return p
end