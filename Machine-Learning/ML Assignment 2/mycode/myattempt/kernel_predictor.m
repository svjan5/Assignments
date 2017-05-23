function y_pred = kernel_predictor(X, filename)
    load(filename, 'model');
    [n,d] = size(X);
    
    
    y_pred = (model.alpha-model.alpha_star)' * compute_kernel(model.X, X, model.kernelType, model.r) + model.b;
    y_pred = y_pred';
end