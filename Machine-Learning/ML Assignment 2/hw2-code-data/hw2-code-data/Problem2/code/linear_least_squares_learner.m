function linear_least_squares_learner(X, y)
    [n,d] = size(X);
    X(:,d+1) = ones(n,1);
    w = inv(X'*X) * X' * y;
    save('least_square_wt', 'w');
end