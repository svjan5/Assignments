function [error, grad] = errorFunction_reg(w, lambda, X, Y)
[m,d] = size(X);


h = sigmoid(X*w);

error = -sum((Y .* log(h))+((1-Y) .* log(1-h))) + (lambda/2 * (w'*w));

grad = zeros(d, 1);

for i = 1:1:d
    grad(i) = sum((h-Y) .* X(:,i)) + lambda ;
end

end