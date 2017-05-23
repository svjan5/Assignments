clear; clc;
load('../data/regression_dataset.mat');

linear_least_squares_learner(train, train_y);
y_pred = linear_predictor(test, 'least_square_wt');

test_error = squared_error(y_pred, test_y')

y_pred = linear_predictor(train, 'least_square_wt');
train_error = squared_error(y_pred, train_y)