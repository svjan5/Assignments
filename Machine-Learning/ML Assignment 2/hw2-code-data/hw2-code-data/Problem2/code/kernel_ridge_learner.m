function kernel_ridge_learner(X, y, lambda, kernelType, r)
    addpath('../../Problem1/code/');
    [n,d] = size(X);
    X(:,d+1) = ones(n,1);
    
    beta = zeros(n,1);
    beta = inv(lambda * eye(n) + compute_kernel(X,X, kernelType, r))*y;
    
    model.kernelType = kernelType;
    model.r = r;
    model.beta = beta;
    model.n = n;
    model.X = X;
    
    save('kernel_regression_model', 'model');
end