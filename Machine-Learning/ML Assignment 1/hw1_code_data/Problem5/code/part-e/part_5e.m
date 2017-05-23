clc; clear;
addpath(genpath('../'));

M = importdata('../../data/train_small.csv');
[m,n] = size(M); n=n-1;

change_count = zeros(20, 1) ;
gamma = zeros(20, 1);
gamma_pred = zeros(20,1);

R = max(diag(M'*M));

for k = 1:1:20
    
    M = shuffle(M);
    X = M;
    Y = X(:,n+1);
    X(:, n+1) = ones(m,1);
    
    w = zeros(n+1,1);
    
    iters = 0;
    count = 0;
    
    while (1)
        count = 0;
        % Updata Model
        for i = 1:1:m
            pred = w'*X(i,:)';
            if( (Y(i) == 1 && pred<0) || (Y(i) == -1 && pred>=0)) 
                w = w + Y(i) * X(i,:)';
                count = count + 1;
            end
        end
    
        if (count == 0)
            break;
        end
        
        iters = iters + 1;
        change_count(k) = change_count(k) + count;
%         fprintf('Iteration: %d, mispredict:%d \n', iters, sum( (Y.*(X*w)) == 1));
    end
    
    
    mul = Y .* (X * w);
    gamma(k) = min(Y .* (X * w));
    
    gamma_pred(k) = R * (w'*w) / change_count(k);
    fprintf('Iteration %d: Change count: %d, Iterations: %d, Gamma: %f , %d\n', k, change_count(k), iters, gamma(k), sum(mul < 0));
end

%% Plot Data
histogram(change_count);
xlabel('no. of changes');
ylabel('Frequency');
title('Problem 5(e)');

plot(1:1:20, gamma, 1:1:20, gamma_pred);
title('Comparing gamma values');
xlabel('Iterations')
ylable('Gamme Value')
    
