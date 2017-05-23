clear; clc;

% train_data = importdata('../data/Synth3/train.txt');
% [m,d] = size(train_data); d = d - 1;
% train_labels = train_data(:, d+1);
% train_data = train_data(:, 1:d);
% 
% test_data = importdata('../data/Synth3/test.txt');
% m_test = size(test_data);
% test_labels = test_data(:, d+1);
% test_data = test_data(:, 1:d);

train_data = importdata('../data/Spambase/train.txt');
[m,d] = size(train_data); d = d - 1;
train_labels = train_data(:, d+1);
train_data = train_data(:, 1:d);

test_data = importdata('../data/Spambase/test.txt');
m_test = size(test_data);
test_labels = test_data(:, d+1);
test_data = test_data(:, 1:d);

svm_model = SVM_learner(train_data, train_labels, 100, 'linear', 0);
svm_predict = SVM_classifier(test_data, svm_model);
svm_err = classification_error(svm_predict, test_labels)

logistic_pred = logisticReg(train_data, train_labels, test_data);

logistic_errs = classification_error(logistic_pred, test_labels)