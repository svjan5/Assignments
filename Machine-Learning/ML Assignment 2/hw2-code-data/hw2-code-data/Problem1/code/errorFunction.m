function [error, grad] = errorFunction(w, X, Y)
[m,d] = size(X);


h = sigmoid(X*w);

error = -sum( (Y .* log(h)) + ((1 - Y) .* log(1 - h)) );

grad = zeros(d, 1);

for i = 1:1:d
    grad(i) = sum((h-Y) .* X(:,i));
end

end