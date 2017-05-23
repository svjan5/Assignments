clear; clc;

C_list = [1, 10, 10^2 , 10^3 , 10^4 ];
K = 5;

train_data = importdata('../data/Spambase/train.txt');
[m,d] = size(train_data); d = d - 1;
train_labels = train_data(:, d+1);
train_data = train_data(:, 1:d);

test_data = importdata('../data/Spambase/test.txt');
m_test = size(test_data);
test_labels = test_data(:, d+1);
test_data = test_data(:, 1:d);

train_error = zeros(length(C_list), K);
test_error = zeros(length(C_list), K);

kfold = cvpartition(m, 'KFold', K);

for i = 1:length(C_list)
    C = C_list(i);
    
    model = SVM_learner(train_data, train_labels, C, 'linear', 0);

    y_pred = SVM_classifier(train_data, model);
    classification_error(y_pred, train_labels)

    y_pred = SVM_classifier(test_data, model);
    classification_error(y_pred, test_labels)
    break;
end

avg_train_error = sum(train_error, 2)/5;
avg_test_error = sum(test_error, 2)/5;

[min_error, minIndex] = min(avg_train_error);

fprintf('Best choice of C is %f', C_list(minIndex));

figure(1);
plot(1:5, avg_train_error);
title('1(a) : Avg cross-validation train error');
ylabel('Error');
xlabel('C');
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});


figure(2);
plot(1:5, avg_test_error);
title('1(a) : Avg cross-validation test error');
ylabel('Error');
xlabel('C');
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});
