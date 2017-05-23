function linear_ridge_learner(X, Y, lambda)

    [n,d] = size(X);
    X(:,d+1) = ones(n,1);
    % w = inv(X'*X) * X' * y;
    w = inv(X'*X + lambda * eye(d+1,d+1)) * X'*Y;
    save('ridge_regression_wt', 'w');
end