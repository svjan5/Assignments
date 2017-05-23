function y_pred = linear_predictor(X, filename)
    [n,d] = size(X);
    X(:,d+1) = ones(n,1);
    load(filename, 'w');
    y_pred = X * w;
end