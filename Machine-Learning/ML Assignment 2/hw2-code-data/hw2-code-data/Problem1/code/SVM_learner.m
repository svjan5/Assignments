function model = SVM_learner(X, y, C, kerneltype, r)
    % INPUT : 
    % traindata   - m X n matrix, where m is the number of training points
    % trainlabels - m X 1 vector of training labels for the training data
    % C           - SVM regularization parameter (positive real number)
    % kerneltype  - one of strings 'linear', 'poly', 'rbf'
    %               corresponding to linear, polynomial, and RBF kernels
    %               respectively.
    % r           - integer parameter indicating the degree of the
    %               polynomial for polynomial kernel, or the width
    %               parameter for the RBF kernel; not used in the case of
    %               linear kerne and can be set to a default value.
    
    % OUTPUT
    % returns the structure 'model' which has the following fields, in
    % addition to the training data/parameters.(You can choose to add more
    % fields to this structure needed for your implementation)
    
    
    % 	alphas      	- m X 1 vector of support vector coefficients
    % 	b           	- SVM bias term
    % 	objective   	- optimal objective value of the SVM solver
    % 	support_vectors - the subset of training data, which are the support vectors
    
    % Default code below. Fill in the code for solving the
    % SVM dual optimization problem using quadprog function
    
    [m,n] = size(X);
    b = 0;
    alpha = zeros(size(X, 1), 1);
    
    K = compute_kernel(X, X, kerneltype, r);
    H = K .* (y*y');
    f = -ones(m,1);
    Aineq = [];
    bineq = [];
    Aeq = y';
    beq = 0;
    
    lb = zeros(m,1);
    ub = ones(m,1)*C;
    
    [alpha, objective] = quadprog(H, f, Aineq, bineq, Aeq, beq, lb, ub);
    
    support_vectors = [];
    supp_y = [];
    w = zeros(n,1);
    
    for i = 1:m
        if alpha(i) > 0
            support_vectors = [support_vectors; X(i,:)];
            supp_y = [supp_y; y(i)];
            w = w + alpha(i) * y(i) * X(i,:)';
        end
    end
    
    count = 0;
    for i = 1:m
        if ( alpha(i) > 0 && alpha(i) < 1 - eps)
%             b = [b; y(i) - X(i,:)*w]
            b_temp = y(i) - alpha'* (y .* compute_kernel(X, X(i,:), kerneltype, r));
            b = b + b_temp;
            count = count  + 1;
        end
    end
    b = b/count;
    
    model.b = b;
    model.objective = objective;
    model.alphas = alpha; 
    model.kerneltype = kerneltype;
    model.r = r;
    model.C = C;
    model.traindata = X;
    model.trainlabels = y;
    model.support_vectors = support_vectors;
    model.support_vectors_y = supp_y;
    
end
