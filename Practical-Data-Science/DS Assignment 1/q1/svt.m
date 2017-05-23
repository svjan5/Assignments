function [Z] = svt(X, Mask) 

addpath('./PROPACK/')
% X = imread('fruit.pgm');
[n1,n2] = size(X);

% sparsity = 0.1;


maxIter = 250;
delta = 1.6;
tau = 15;

Y = zeros(n1,n2);
Z = zeros(n1,n2);

for k = 1:maxIter
    [U,S,V] = svd(Y, 'econ');
    S_hat = max(S - eye(size(S,1)) * tau, 0);
    Z = U * S_hat * V';
    Y = Y + delta * (Mask .* (X - Z) );    
end

% subplot(1,2,1); imshow(X)
% subplot(1,2,2); imshow(Z)