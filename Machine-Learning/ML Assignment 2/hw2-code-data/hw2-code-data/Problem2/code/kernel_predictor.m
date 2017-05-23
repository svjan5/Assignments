function y_pred = kernel_predictor(X, filename)
    load(filename, 'model');
    [n2,d] = size(X);
    X(:,d+1) = ones(n2,1);
    y_pred = zeros(n2,1);
    
    y_pred = model.beta' * compute_kernel(model.X, X, model.kernelType, model.r);
    y_pred = y_pred';
%     for i = 1:n2
%         for j = 1:model.n;
%             y_pred(i) = y_pred(i) + model.beta(j) * compute_kernel( model.X(j,:), X(i,:), model.kernelType, model.r);
%         end
%     end
end