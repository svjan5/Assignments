X = importdata('data.csv');
[m,n] = size(X);
% plot(X(:,1), X(:,2), '.')

D = zeros(m,m);

for i = 1:1:m
    for j = 1:1:m
        D(i,j) = X(i,1) - X
    end
end