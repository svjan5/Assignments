clear; clc;

% Finding the best value of lambda
load('../data/regression_folds.mat');

lambda_list = [0.01, 0.1, 1, 10, 100];
test_error = zeros(length(lambda_list), 5);
train_error = zeros(length(lambda_list), 5);

for i = 1:5
    lambda = lambda_list(i);
    for k = 1:5
        train_data(:,:) = fold_train(k,:,:);
        train_labels(:, 1) = fold_train_y(k,:);
        test_data(:, :) = fold_test(k,:,:);
        test_labels(:,1) = fold_test_y(k,:);
        
        kernel_ridge_learner(train_data, train_labels, lambda, 'poly', 3);
        
        y_pred = kernel_predictor(train_data, 'kernel_regression_model');
        train_error(i,k) = squared_error(y_pred, train_labels);
        
        y_pred = kernel_predictor(test_data, 'kernel_regression_model');
        test_error(i,k) = squared_error(y_pred, test_labels);
    end
end

test_error
train_error


%% Plot
avg_test_error = sum(test_error,2)/5
avg_train_error = sum(train_error,2)/5

H = plot(1:5, avg_train_error, 1:5, avg_test_error);
set(gca,'Xtick',1:5,'XTickLabel', {'0.01', '0.1', '1', '10', '100'});
legend('Avg train error', 'Avg test error', 'Location','northwest')
title('2(e) : Average cross-validation Error');
ylabel('Average Cross-validation error');
xlabel('lambda');
% saveas(H, '../plot/2e_avg_error', 'png')

% [~, ind] = min(avg_error);
% best_lambda = lambda_list(ind);

%% Testing on entire data set 
load('../data/regression_dataset.mat');
lambda_list = [0.01, 0.1, 1, 10, 100];

full_train_error = zeros(length(lambda_list), 1);
full_test_error = zeros(length(lambda_list), 1);

for i = 1:length(lambda_list)
    lambda = lambda_list(i);
    
    kernel_ridge_learner(train, train_y, lambda, 'poly', 3);
    
    y_pred = kernel_predictor(train, 'kernel_regression_model');
    full_train_error(i) = squared_error(y_pred, train_y);
    
    y_pred = kernel_predictor(test, 'kernel_regression_model');
    full_test_error(i) = squared_error(y_pred, test_y');
end


full_train_error
full_test_error
