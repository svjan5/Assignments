%% Training Model
clc; 
clear; X = csvread('../../data/train_small.csv');

[m,n] = size(X); n=n-1;
Y = X(:, n+1);
X = X(:, 1:1:n);

numClass = 2;

% Prior Probabilities
prior = zeros(numClass, 1);

prior(1) = sum(Y == 1) / m;
prior(2) = sum(Y == -1) / m;

% Class-conditional densities
condprob = zeros(n, numClass);

M_pos = X(Y ==  1, :);
M_neg = X(Y == -1, :);

m1 = sum(sum(M_pos));
m2 = sum(sum(M_neg));

for i = 1:1:n
    condprob(i, 1) = (sum(M_pos(:,i)) + 1) / (m1+n);
    condprob(i, 2) = (sum(M_neg(:,i)) + 1) / (m2+n);
end

fprintf('Training Completed');
%% Calculate Training Accuracy
clc;

X = csvread('../../data/train_small.csv');
[m,n] = size(X); 
Y = X(:, n); 
X = X(:, 1:1:n-1);
n = n - 1;
pred = zeros(m,1);

for i = 1:1:m
    score = zeros(numClass, 1);
    for c = 1:1:numClass
        score(c) = log(prior(c));
        for j = 1:1:n
            score(c) = score(c) + X(i,j) * log(condprob(j,c));
        end 
    end
    if(score(1) > score(2))
        pred(i) = 1;
    else
        pred(i) = -1;
    end
end

accuracy_training = (sum(Y == pred) / m)*100;
fprintf('Training Accuracy: %f \n', accuracy_training);

%% Calculate Test Accuracy
clc;
X_test = csvread('../../data/test.csv');

m_test = size(X_test,1);
Y_test = X_test(:,n+1);

pred = zeros(m_test,1);

for i = 1:1:m_test
    score = zeros(numClass, 1);
    for c = 1:1:numClass
        score(c) = log(prior(c));
        for j = 1:1:n
            score(c) = score(c) + X_test(i,j) * log(condprob(j,c));
        end 
    end
    if(score(1) > score(2))
        pred(i) = 1;
    else
        pred(i) = -1;
    end
end

accuracy_test = (sum(Y_test == pred) / m_test) * 100;
fprintf('Testing Accuracy: %f \n', accuracy_test);
confusion_matrix = confusionmat(Y_test, pred)