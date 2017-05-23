function [W,H] = nmr_3(X, num, maxIters, tau, is_nndsvd)
[m,n] = size(X);

if (is_nndsvd == 0)
    W = rand(m,num);
    H = rand(num,n);    
else
    [W,H] = NNDSVD(X,num,0);
end

counter = 1;
while(1)
    wh = W*H;

    for i = 1:m
        for a = 1:num
            term_num = sum(H(a,:) .* X(i,:) ./ wh(i,:));
            term_den = sum(H(a,:)) + tau * W(i,a);
            W(i,a) = W(i,a) * term_num / term_den;
        end
    end

    wh = W*H;
    for a = 1:num
        for u = 1:n
            term_num = sum(W(:,a) .* X(:,u) ./ wh(:,u));
            term_den = sum(W(:,a)) + tau * H(a,u);
            H(a,u) = H(a,u) * term_num / term_den;
        end
    end

%     App_X = W*H;
%     diff = norm(X- App_X, 'fro');
%     fprintf('Iteration: %d, Frobenius: %f \n', counter, diff);
    fprintf('.');
    counter = counter + 1;
    
    if(counter >= maxIters)
        break;
    end
end

fprintf('\n');