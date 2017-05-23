function [V] = isvd(X, Mask)

% X = imread('fruit.pgm');
[n,m] = size(X);

d = m;

U = orth(rand(n,d));
W = orth(randn(d,d));
V = ones(n,m);

for t = 1:m
    omega = Mask(:,t);
    v = X(:,t);
    U_omega = Mask .* U;
    w = U_omega' * v;

    p = (U*w);
    for i = 1:n
        if (omega(i) == 1)
            V(i,t) = v(i);
        else
            V(i,t) = p(i) ;
        end
    end

    r = V(:,t) - p;

    Upt = eye(d+1);
    Upt(1:d,d+1) = w;
    Upt(d+1,d+1) = norm(r);

    [U1, S1, V1] = svd(Upt);
    U_cap = U1(:,1:d);

    U = [U r/norm(r)] * U_cap * W;
end

% subplot(1,2,1); imshow(X)
% subplot(1,2,2); imshow(V);