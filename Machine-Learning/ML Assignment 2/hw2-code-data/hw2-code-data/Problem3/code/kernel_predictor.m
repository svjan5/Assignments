function y = kernel_predictor(x, filename)
    % INPUT
    % testdata - m X n matrix of the test data samples
    % model    - SVM model structure returned by SVM_learner
    
    % OUTPUT
    % labels - m x 1 vector of predicted labels
    
    % Write code here    
    
    load(filename, 'model');
    alphas = model.alphas ;
    b = model.b;
    kerneltype = model.kerneltype;
    r = model.r;
    traindata = model.X;
    
    y = alphas'*compute_kernel(traindata,x,kerneltype,r) + b ;
    y = y';
   
end