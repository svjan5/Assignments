clc;clear; 
addpath(genpath('../'));

lambdas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1];
fold_label = {'Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5'};

error_train = zeros(5, length(lambdas));
error_test  = zeros(5, length(lambdas));

for fold = 1:1:5
    
    train_loc = strcat('../../data/spambase-cross-validation/', fold_label(fold), '/cv-train.txt');
    test_loc  = strcat('../../data/spambase-cross-validation/', fold_label(fold), '/cv-test.txt');
    
    train_loc = train_loc{1};
    test_loc = test_loc{1};
    
    % Load Training Data
    X = importdata(train_loc);
    [m,d] = size(X); d=d-1;         % Not considering the last column
    Y = X(:, d+1); Y(Y == -1) = 0;  % Replacing all -1 with 0 
    X(:, d+1) = ones(m,1);          % Replacing last columns with all ones
    
    % Load Test Data
    X_test = importdata(test_loc);
    m_test = size(X_test, 1);
    Y_test = X_test(:, d+1); Y_test(Y_test == -1) = 0;
    X_test(:, d+1) = ones(m_test,1);

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

        error_train(fold, k) = classification_error(pred_train, Y);

        % Predict on Testing data
        pred_test  = zeros(m_test, 1);
        for i = 1:1:m_test
            if (w'*X_test(i,:)' > 0)
                pred_test(i) = 1;
            else
                pred_test(i) = 0;
            end
        end

        error_test(fold, k) = classification_error(pred_test, Y_test);

        fprintf('Fold: %d: Using %f lambda value, Training: %f, Testing: %f\n', fold, lambda, error_train(fold, k), error_test(fold, k));
    end
end

sum(error_train)/5
sum(error_test)/5

%% Plot Values
plot(1:1:length(lambdas), error_test);
set(gca,'Xtick',1:8,'XTickLabel', {'1e-7', '1e-6', '1e-5', '1e-4', '1e-3', '1e-2', '1e-1', '1'})
title('Problem 4(b) Cross-validation');
xlabel('Lambda');
ylabel('Error');
legend('Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5')