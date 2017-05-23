%% Load Datasets
clc; clear;
addpath(genpath('../'));

% Load Small Training Data
X_small = importdata('../../data/train_small.csv');
[m_small,n] = size(X_small); n=n-1;
Y_small = X_small(:, n+1);
X_small(:, n+1) = ones(m_small,1);

% Load medium Training Data
X_medium = importdata('../../data/train_medium.csv');
m_medium = size(X_medium,1);
Y_medium = X_medium(:, n+1);
X_medium(:, n+1) = ones(m_medium,1);

% Load large Training Data
X_large = importdata('../../data/train_large.csv');
m_large = size(X_large,1);
Y_large = X_large(:, n+1);
X_large(:, n+1) = ones(m_large,1);

% Load Testing Data
X_test = csvread('../../data/test.csv');
m_test = size(X_test, 1); 
Y_test = X_test(:, n+1);
X_test(:, n+1) = ones(m_test,1);


%% Train Small model
w_small = zeros(n+1,1);
training_time = zeros(3,1);

tic();
while (1)
    count = 0;
    
    for i = 1:1:m_small
        pred = w_small'*X_small(i,:)';
        if( (Y_small(i) == 1 && pred<0) || (Y_small(i) == -1 && pred>=0)) 
            w_small = w_small + Y_small(i) * X_small(i,:)';
            count = count + 1;
        end
        
    end
    if (count == 0)
        break;
    end
end
training_time(1) = toc();

%% Train Medium model
w_medium = zeros(n+1,1);

tic();
while (1)
    count = 0;
    
    for i = 1:1:m_medium
        pred = w_medium'*X_medium(i,:)';
        if( (Y_medium(i) == 1 && pred<0) || (Y_medium(i) == -1 && pred>=0)) 
            w_medium = w_medium + Y_medium(i) * X_medium(i,:)';
            count = count + 1;
        end
    end

    if (count == 0)
        break;
    end
end
training_time(2) = toc();

%% Train Large model
w_large = zeros(n+1,1);
tic();
while (1)
    count = 0;
    
    for i = 1:1:m_large
        pred = w_large'*X_large(i,:)';
        if( (Y_large(i) == 1 && pred<0) || (Y_large(i) == -1 && pred>=0)) 
            w_large = w_large + Y_large(i) * X_large(i,:)';
            count = count + 1;
        end
    end

    if (count == 0)
        break;
    end
end
training_time(3) = toc();

%% Calculate Test Accuracy
acc_test = zeros(3,1);

% Predict on Test Data
pred_test = zeros(m_test, 1);
for i = 1:1:m_test
    if( w_small'*X_test(i,:)' < 0)
        pred_test(i) = -1;
    else
        pred_test(i) = 1;
    end
end
acc_test(1) = Classification_Accuracy(pred_test, Y_test);

% Predict on Test Data
for i = 1:1:m_test
    if( w_medium'*X_test(i,:)' < 0)
        pred_test(i) = -1;
    else
        pred_test(i) = 1;
    end
end
acc_test(2) = Classification_Accuracy(pred_test, Y_test);

% Predict on Test Data
for i = 1:1:m_test
    if( w_large'*X_test(i,:)' < 0)
        pred_test(i) = -1;
    else
        pred_test(i) = 1;
    end
end
acc_test(3) = Classification_Accuracy(pred_test, Y_test);

%% Plot
plot([m_small, m_medium, m_large], training_time);
xlabel('Number of training examples');
ylabel('Training Time');
title('Problem 5(g)');