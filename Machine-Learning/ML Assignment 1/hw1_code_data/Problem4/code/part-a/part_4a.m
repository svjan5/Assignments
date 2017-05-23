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

error_train = zeros(10, 1);
acc_test  = zeros(10, 1);
index = 1:1:m;

for k = 1:1:10
    m_train = int16(k*0.1*m);
%     rand_index = datasample(index, m_train, 2, 'Replace', false); 
    X_train = X(1:1:m_train, :);
    Y_train = Y(1:1:m_train);
    
    w = zeros(d+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 2000, 'MaxFunEvals', 10000);
    [w  cost] = fminunc(@(t)(errorFunction(t, X_train, Y_train)), w, options);
    
    fprintf('Cost at theta found by fminunc: %f\n', cost);
    
    % Predict on Training Data
    pred_train = zeros(m_train,1);
    for i = 1:1:m_train
        if (w'*X_train(i,:)' > 0)
            pred_train(i) = 1;
        else
            pred_train(i) = 0;
        end
    end
    
    error_train(k) = classification_error(pred_train, Y_train);
    
    % Predict on Testing data
    pred_test  = zeros(m_test, 1);
    for i = 1:1:m_test
        if (w'*X_test(i,:)' > 0)
            pred_test(i) = 1;
        else
            pred_test(i) = 0;
        end
    end
    
    acc_test(k) = classification_error(pred_test, Y_test);
    
    fprintf('Using %d%% data the accuracy on Training: %f, Testing: %f', k*10, error_train(k), acc_test(k));
end

plot(10:10:100, error_train, 10:10:100, acc_test);
title('Problem 4(a)');
xlabel('Percentage of Data used');
ylabel('Error');
legend('Training', 'Testing')