clc; clear;
addpath(genpath('../'));

% Load Training Data
X = importdata('../../data/train_small.csv');
[m,n] = size(X); n=n-1;
Y = X(:, n+1);
X(:, n+1) = ones(m,1);

% Load Testing Data
X_test = csvread('../../data/test.csv');
m_test = size(X_test, 1); 
Y_test = X_test(:, n+1);
X_test(:, n+1) = ones(m_test,1);

gamma = zeros(20, 1);
etas = 0.05:0.05:0.6;

acc_train    = zeros(length(etas), 1);
acc_test     = zeros(length(etas), 1);
change_count = zeros(length(etas), 1) ;

% Training Perceptron for different values of 
for k = 1:1:length(etas);
    
    eta = etas(k);
    w = zeros(n+1,1);
    
    while (1)
        count = 0;
        % Updata Model
        for i = 1:1:m
            pred = w'*X(i,:)';
            if( (Y(i) == 1 && pred<0) || (Y(i) == -1 && pred>=0)) 
                w = w + eta * Y(i) * X(i,:)';
                count = count + 1;
            end
        end
    
        if (count == 0)
            break;
        end
        
        change_count(k) = change_count(k) + count;
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

    fprintf('Eta: %f, Update count: %d, Training Accuracy: %f, Test Accuracy: %f\n', eta, change_count(k), acc_train(k), acc_test(k));
end

%% Plot Data
plot(etas, change_count);
xlabel('\eta', 'interpreter', 'tex');
ylabel('No of updates');
title('Problem 5(f)');