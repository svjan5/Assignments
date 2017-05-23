function [W,H] = nmr_2(X, num, maxIters, tau, is_nndsvd)
[m,n] = size(X);

if (is_nndsvd == 0)
    W = rand(m,num);
    H = rand(num,n);    
else
    [W,H] = NNDSVD(X,num,0);
end
    
eta_w = zeros(size(W));
eta_h = zeros(size(H));
one_w = ones(size(W));
one_h = ones(size(H));

counter = 1;
while(1)
    den_w = W*(H*H') + tau*one_w;

    for i = 1:m
        for j = 1:num
            eta_w(i,j) = W(i,j) / den_w(i,j);
        end
    end

    W = eta_w .* (X*H');

    den_h = (W'*W)*H + tau*one_h;

    for i = 1:num
        for j = 1:n
            eta_h(i,j) = H(i,j) / den_h(i,j);
        end
    end

    H = eta_h .* (W'*X);

%     App_X = W*H;
%     diff = norm(X- App_X, 'fro');
    fprintf('.', counter);
    counter = counter + 1;
    if(counter >= maxIters)
        break;
    end
end
fprintf('\n');