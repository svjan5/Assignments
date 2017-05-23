function SVR_learner(X, y, C, kerneltype, r, epsilon)

n = size(X,1);
K = compute_kernel(X, X, kerneltype, r);
H = [K, -K; -K, K];
f = [epsilon * ones(n,1) - y; epsilon * ones(n,1) + y];
Aeq = [ones(n,1); -ones(n,1)]';
beq = 0;
lb = zeros(2*n,1);
ub = C * ones(2*n,1);

[alphaTemp, objective] = quadprog(H, f, [], [], Aeq, beq, lb, ub);

tol = 10^-7;
alpha = alphaTemp(1:n,:);
alpha_star = alphaTemp(n+1: 2*n);
alphas = alpha - alpha_star;

b=0; count=0;

for i=1:n
    if (alpha(i) >= tol && alpha(i) <= (C-tol)) || (alpha_star(i) >= tol && alpha_star(i) <= (C-tol))
        if (alpha(i) >= tol) && (alpha(i)<=(C-tol))
            b = b + y(i) - alphas'*compute_kernel(X,X(i,:),kerneltype,r) - epsilon;       
        else
            b = b - y(i) + alphas'*compute_kernel(X,X(i,:),kerneltype,r)  - epsilon;
        end
        count=count+1;
    end
end
b=b/count;

model.b = b;
model.objective = -objective;
model.alphas = alphas;
model.kerneltype = kerneltype;
model.r = r;
model.C = C;
model.X = X;
model.y = y;
model.support_vectors = X;

save('svr_model', 'model');

end
