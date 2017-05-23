clc; clear;

% For Classification_Accuracy function
addpath(genpath('../'))

% Load Training Data
X = csvread('../../data/train_small.csv');
[m,n] = size(X); n=n-1;
Y = X(:, n+1);

% Load Testing Data
X_test = csvread('../../data/test.csv');
m_test = size(X_test, 1); 
Y_test = X_test(:, n+1);


X = sign(X);
X_test = sign(X_test);

X_test(:, n+1) = -1*ones(m_test,1);
X(:, n+1) = -1*ones(m,1);

% scale_factor = max (max(X), max(X_test));
% for i = 1:1:n
%     if(scale_factor(i) ~= 0)
%         X(:, i) = X(:,i) / scale_factor(i);
%         X_test(:, i) = X_test(:,i) / scale_factor(i);
%     end
% end

Iters = 25;
w = ones(n+1,1)/(n+1);
theta = 1/(n+1);
theta = 0;

acc_train = zeros(Iters, 1);
acc_test  = zeros(Iters, 1);
eta = 0.3;

% Iterations 1 to 25 
for k = 1:1:Iters
    count = 0;

    % Updata weights
    for i = 1:1:m
        pred = w'*X(i,:)';
        if( (Y(i) == 1 && pred < theta) || (Y(i) == -1 && pred >= theta)) 
            
            Z = 0;
            for j = 1:1:n+1
                Z = Z + w(j)*exp(eta * Y(i) * X(i,j));
            end
            
            for j = 1:1:n+1
                w(j) = w(j)*exp(eta*Y(i)*X(i,j));
            end
            
            w = w/Z;
            count = count + 1;
        end
    end
    
    % Predict on Training Data
    pred = zeros(m, 1);
    for i = 1:1:m
        if( w'*X(i,:)' < theta)
            pred(i) = -1;
        else
            pred(i) = 1;
        end
    end
    
    % Predict on Test Data
    pred_test = zeros(m_test, 1);
    for i = 1:1:m_test
        if( w'*X_test(i,:)' < theta)
            pred_test(i) = -1;
        else
            pred_test(i) = 1;
        end
    end
    
    acc_train(k) = Classification_Accuracy(pred, Y);
    acc_test(k)  = Classification_Accuracy(pred_test, Y_test);
    
    fprintf('Round %d: Training Accuracy: %f, Testing Accuracy: %f \n', k, acc_train(k), acc_test(k));
end

plot(1:1:k, acc_train, 1:1:k, acc_test);
title('Problem 5(c)');
xlabel('Rounds');
ylabel('Accuracy');
legend('Training', 'Testing')