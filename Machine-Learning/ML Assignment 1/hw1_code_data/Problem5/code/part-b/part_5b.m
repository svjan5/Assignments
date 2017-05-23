clc; clear;

% For Classification_Accuracy function
addpath(genpath('../'))

% Load Training Data
X = csvread('../../data/train_small.csv');
[m,n] = size(X); n=n-1;
Y = X(:, n+1);
X(:, n+1) = ones(m,1);

% Load Testing Data
X_test = csvread('../../data/test.csv');
m_test = size(X_test, 1); 
Y_test = X_test(:, n+1);
X_test(:, n+1) = ones(m_test,1);

Iters = 25;
w = zeros(n+1,1);

acc_train = zeros(Iters, 1);
acc_test  = zeros(Iters, 1);

% Iterations 1 to 25 
for k = 1:1:Iters
    count = 0;
    
    % Updata weights
    for i = 1:1:m
        pred = w'*X(i,:)';
        if( (Y(i) == 1 && pred<0) || (Y(i) == -1 && pred>=0)) 
            w = w + Y(i) * X(i,:)';
            count = count + 1;
        end
    end
    
    % Predict on Training Data
    pred = zeros(m, 1);
    for i = 1:1:m
        if( w'*X(i,:)' < 0)
            pred(i) = -1;
        else
            pred(i) = 1;
        end
    end
    
    % Predict on Test Data
    pred_test = zeros(m_test, 1);
    for i = 1:1:m_test
        if( w'*X_test(i,:)' < 0)
            pred_test(i) = -1;
        else
            pred_test(i) = 1;
        end
    end
    
    acc_train(k) = Classification_Accuracy(pred, Y);
    acc_test(k)  = Classification_Accuracy(pred_test, Y_test);
    
    fprintf('Round %d: Training Accuracy: %f, Testing Accuracy: %f \n', k, acc_train(k), acc_test(k));
end

%%
plot(1:1:k, acc_train, 1:1:k, acc_test);
title('Problem 5(b)');
xlabel('Rounds');
ylabel('Accuracy');
legend('Training', 'Testing')