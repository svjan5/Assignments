clc;clear;

load('../data/regression_folds.mat');

% Finding the best value of lambda
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
        
        linear_ridge_learner(train_data, train_labels, lambda);
        
        y_pred = linear_predictor(train_data, 'ridge_regression_wt');
        train_error(i,k) = squared_error(y_pred, train_labels);
        
        y_pred = linear_predictor(test_data, 'ridge_regression_wt');
        test_error(i,k) = squared_error(y_pred, test_labels);
    end
end

test_error
train_error
% test_error = test_error / 5
% train_error = train_error / 5

avg_test_error = sum(test_error,2)/5
avg_train_error = sum(train_error,2)/5

H = plot(1:5, avg_train_error, 1:5, avg_test_error);
set(gca,'Xtick',1:5,'XTickLabel', {'0.01', '0.1', '1', '10', '100'});
legend('Avg train error', 'Avg test error', 'Location','northwest')
title('2(d) : Average cross-validation Error');
ylabel('Average Cross-validation error');
xlabel('lambda');
% saveas(H, '../plot/2d_avg_error', 'png')

% [~, ind] = min(avg_error);
% best_lambda = lambda_list(ind);


%% Finding error on complete train and test dataset for all lambdas
clc
load('../data/regression_dataset');
train_error = zeros(length(lambda_list), 1);
test_error = zeros(length(lambda_list), 1);

for i = 1:length(lambda_list)
    lambda = lambda_list(i);
    
    linear_ridge_learner(train, train_y, lambda);
    y_pred_train = linear_predictor(train, 'ridge_regression_wt');
    train_error(i) = squared_error(y_pred_train, train_y);

    y_pred_test = linear_predictor(test, 'ridge_regression_wt');
    test_error(i) = squared_error(y_pred_test, test_y');
end

train_error
test_error
