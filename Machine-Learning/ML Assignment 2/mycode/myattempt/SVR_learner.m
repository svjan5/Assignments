function SVR_learner(X, y, C, epsilon, tol, kernelType, r)
addpath('../../Problem1/code');
[n,d] = size(X);

K = compute_kernel(X, X, kernelType, r);
H = [K, -K; -K, K];
f = [epsilon * ones(n,1) - y; epsilon * ones(n,1) + y];
Aineq = [];
bineq = [];
Aeq = [ones(n,1); -ones(n,1)]';
beq = 0;
lb = zeros(2*n,1);
ub = C * ones(2*n,1);

[alphas, objective] = quadprog(H, f, Aineq, bineq, Aeq, beq, lb, ub);

alpha      = alphas(1:n);
alpha_star = alphas(n+1:2*n);

% Compute w
% tol = 0.1;
w = 0;
ind = zeros(n,1);
% for i = 1:n
%     if ( (alpha(i) >= C-tol && alpha(i) <= C + tol) || (alpha_star(i) >= C-tol && alpha_star(i) <= C + tol)) || ((alpha(i) > tol && alpha(i) < C - tol) || (alpha_star(i) > tol && alpha_star(i) < C - tol)) 
% %         w = w + (alpha(i) - alpha_star(i))*X(i,:);
%         ind(i) = 1;
%     end
% end

beta = alpha - alpha_star;

% Compute b
b = 0; count = 0;
% for i = 1:n
%     if (alpha(i) > tol) && (alpha(i) < C - tol)
% %         b = b + y(i) - X(i,:)*w' - epsilon;
%         b = b + y(i) - (beta .* ind)' * K(:,i)  - epsilon;
%         count = count+1;
%     end
%     if (alpha_star(i) > tol) && (alpha_star(i) < C - tol)
%         b = b + -y(i) + (beta .* ind)' * K(:,i)- epsilon;
%         count = count+1;
%     end
% end
for i=1:n
    if (alpha(i)>=tol && alpha(i)<=(C-tol)) || (alpha_star(i)>=tol && alpha_star(i)<=(C-tol))
        count = count+1;
        if alpha(i)>=tol && alpha(i)<=(C-tol)
            b = b + y(i) - beta'*compute_kernel(X,X(i,:),kernelType,r) - epsilon;       
        else
            b = b - y(i) + beta'*compute_kernel(X,X(i,:),kernelType,r)  - epsilon;
        end
    end
end

b = b/count;
model.b = b;
model.alpha = alpha;
model.alpha_star = alpha_star;
model.X = X;
model.kernelType = kernelType;
model.r = r;
model.objective = objective;

save('svr_model', 'model');