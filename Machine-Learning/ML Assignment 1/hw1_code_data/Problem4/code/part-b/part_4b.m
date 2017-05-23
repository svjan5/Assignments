%% Load Data
clc;clear;

addpath(genpath('../'));
% Training Data
X = importdata('../../data/train.txt');
[m,d] = size(X); d=d-1;         % Not considering the last column
Y = X(:, d+1); Y(Y == -1) = 0;  % Replacing all -1 with 0 
X(:, d+1) = ones(m,1);          % Replacing last columns with all ones

% Test Data
X_test = importdata('../../data/test.txt');
m_test = size(X_test, 1);
Y_test = X_test(:, d+1); Y_test(Y_test == -1) = 0;
X_test(:, d+1) = ones(m_test,1);

lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];

% Train Model

error_train = zeros(length(lambdas), 1);
error_test  = zeros(length(lambdas), 1);

for k = 1:1:length(lambdas)
    lambda = lambdas(k);
    
    
    w = zeros(d+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 400);
    w = fminunc(@(t)(errorFunction_reg(t, lambda, X, Y)), w, options);
    
    % Predict on Training Data
    pred_train = zeros(m,1);
    for i = 1:1:m
        if (w'*X(i,:)' > 0)
            pred_train(i) = 1;
        else
            pred_train(i) = 0;
        end
    end
    
    error_train(k) = classification_error(pred_train, Y);
    
    % Predict on Testing data
    pred_test  = zeros(m_test, 1);
    for i = 1:1:m_test
        if (w'*X_test(i,:)' > 0)
            pred_test(i) = 1;
        else
            pred_test(i) = 0;
        end
    end
    
    error_test(k) = classification_error(pred_test, Y_test);
    fprintf('Using %f lambda value, Training: %f, Testing: %f', lambda, error_train(k), error_test(k));
end


%% Plot Values
plot(1:1:length(lambdas), error_train, 1:1:length(lambdas), error_test);
set(gca,'Xtick',1:8,'XTickLabel', {'1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1'})
title('Problem 4(b)');
xlabel('Lambda');
ylabel('Error');
legend('Training', 'Testing')