clear; clc;

C_list = [1; 10; 10^2; 10^3; 10^4 ];
K = 5;

train_data = importdata('../data/Spambase/train.txt');
[m,d] = size(train_data); d = d - 1;
train_labels = train_data(:, d+1);
train_data = train_data(:, 1:d);

test_data = importdata('../data/Spambase/test.txt');
m_test = size(test_data);
test_labels = test_data(:, d+1);
test_data = test_data(:, 1:d);

fold_train_error = zeros(length(C_list), 1);
fold_test_error = zeros(length(C_list), 1);
test_error = zeros(length(C_list), 1);

kfold = cvpartition(m, 'KFold', K);

for i = 1:length(C_list)
    C = C_list(i);
    
    for k = 1:5
        X = train_data(kfold.training(k), :);
        y = train_labels(kfold.training(k));
        
        X_test = train_data(kfold.test(k), :);
        y_test = train_labels(kfold.test(k));

        model = SVM_learner(X, y, C, 'linear', 0);
        
        y_pred = SVM_classifier(X, model);
        fold_train_error(i) = fold_train_error(i) + classification_error(y_pred, y);
        
        y_pred = SVM_classifier(X_test, model);
        fold_test_error(i) = fold_test_error(i) + classification_error(y_pred, y_test);
        
        y_pred = SVM_classifier(test_data, model);
        test_error(i) = test_error(i) + classification_error(y_pred, test_labels);
    end
end

fold_train_error = fold_train_error / 5;
fold_test_error = fold_test_error / 5;
test_error = test_error / 5;

%% Plot
figure(1);

plot(1:5, fold_train_error); hold on;
H = plot(1:5, fold_test_error);
% plot(1:5, test_error);
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});
legend('Avg train error', 'Avg test error')
title('1(a) : Average cross-validation Error');
ylabel('Average Cross-validation error');
xlabel('C');
saveas(H, '../plot/1a.png', 'png')
