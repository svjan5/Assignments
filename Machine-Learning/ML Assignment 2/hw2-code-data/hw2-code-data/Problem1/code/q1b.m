%% Initialize variables

clear; clc;

C_list = [1; 10; 10^2; 10^3; 10^4 ];
K = 5;

train_data = importdata('../data/Synth3/train.txt');
[m,d] = size(train_data); d = d - 1;
train_labels = train_data(:, d+1);
train_data = train_data(:, 1:d);

test_data = importdata('../data/Synth3/test.txt');
m_test = size(test_data);
test_labels = test_data(:, d+1);
test_data = test_data(:, 1:d);
seedm(1234)
kfold = cvpartition(m, 'KFold', K);

%% Using Linear Kernel
linear_train_error = zeros(length(C_list), 1);
linear_test_error = zeros(length(C_list), 1);

for i = 1:length(C_list)
    C = C_list(i);
    
    for k = 1:5
        X = train_data(kfold.training(k), :);
        y = train_labels(kfold.training(k));
        
        X_test = train_data(kfold.test(k), :);
        y_test = train_labels(kfold.test(k));

        model = SVM_learner(X, y, C, 'linear', 0);
        
        y_pred = SVM_classifier(X, model);
        linear_train_error(i) = linear_train_error(i) + classification_error(y_pred, y);
        
        y_pred = SVM_classifier(X_test, model);
        linear_test_error(i) = linear_test_error(i) + classification_error(y_pred, y_test);
    end
end

linear_train_error = linear_train_error / 5;
linear_test_error = linear_test_error / 5;

%% Using polynomial degree 2 kernel

pdeg2_train_error = zeros(length(C_list), 1);
pdeg2_test_error = zeros(length(C_list), 1);

for i = 1:length(C_list)
    C = C_list(i);
    
    for k = 1:5
        X = train_data(kfold.training(k), :);
        y = train_labels(kfold.training(k));
        
        X_test = train_data(kfold.test(k), :);
        y_test = train_labels(kfold.test(k));

        model = SVM_learner(X, y, C, 'poly', 2);
        
        y_pred = SVM_classifier(X, model);
        pdeg2_train_error(i) = pdeg2_train_error(i) + classification_error(y_pred, y);
        
        y_pred = SVM_classifier(X_test, model);
        pdeg2_test_error(i) = pdeg2_test_error(i) + classification_error(y_pred, y_test);
        
    end
end

pdeg2_train_error = pdeg2_train_error / 5;
pdeg2_test_error = pdeg2_test_error / 5;

%% Using polynomial degree 3 kernel


pdeg3_train_error = zeros(length(C_list), 1);
pdeg3_test_error = zeros(length(C_list), 1);

for i = 1:length(C_list)
    C = C_list(i);
    
    for k = 1:5
        X = train_data(kfold.training(k), :);
        y = train_labels(kfold.training(k));
        
        X_test = train_data(kfold.test(k), :);
        y_test = train_labels(kfold.test(k));

        model = SVM_learner(X, y, C, 'poly', 3);
        
        y_pred = SVM_classifier(X, model);
        pdeg3_train_error(i) = pdeg3_train_error(i) + classification_error(y_pred, y);
        
        y_pred = SVM_classifier(X_test, model);
        pdeg3_test_error(i) = pdeg3_test_error(i) + classification_error(y_pred, y_test);
        
    end
end

pdeg3_train_error = pdeg3_train_error / 5;
pdeg3_test_error = pdeg3_test_error / 5;

%% Using RBF kernel

sig_list = [1/32, 1/4, 1, 4, 32];

rbf_train_error = zeros(length(C_list), length(sig_list));
rbf_test_error = zeros(length(C_list), length(sig_list));

for i = 1:length(C_list)
    C = C_list(i);
    
    for j = 1:length(sig_list)    
        sig = sig_list(j);

        for k = 1:5
            X = train_data(kfold.training(k), :);
            y = train_labels(kfold.training(k));

            X_test = train_data(kfold.test(k), :);
            y_test = train_labels(kfold.test(k));

            model = SVM_learner(X, y, C, 'rbf', sig);

            y_pred = SVM_classifier(X, model);
            rbf_train_error(i,j) = rbf_train_error(i,j) + classification_error(y_pred, y);

            y_pred = SVM_classifier(X_test, model);
            rbf_test_error(i,j) = rbf_test_error(i,j) + classification_error(y_pred, y_test);
        end
    end
end

%% Plot linear
figure(1);

plot(1:5, linear_train_error); hold on;
plot(1:5, linear_test_error);
% plot(1:5, test_error);
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});

legend('Avg train error', 'Avg test error')
title('1(b) : Average cross-validation Error - Linear');
ylabel('Average Cross-validation error');
xlabel('C');
saveas(H, '../plot/1b_linear.png', 'png')

linear_train_error
linear_test_error

%% Plot section pdeg2
figure(1)

plot(1:5, pdeg2_train_error); hold on;
H = plot(1:5, pdeg2_test_error);
% plot(1:5, test_error);
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});
% axis([1,5,0.098,.114])
legend('Avg train error', 'Avg test error', 'Location','northwest')
title('1(b) : Average cross-validation Error - Poly deg 2');
ylabel('Average Cross-validation error');
xlabel('C');
saveas(H, '../plot/1b_pdeg2.png', 'png')

pdeg2_train_error
pdeg2_test_error

%% Plot section pdeg3
figure(1)

plot(1:5, pdeg3_train_error); hold on;
H = plot(1:5, pdeg3_test_error);
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});
% axis([1,5,0.098,.114])
legend('Avg train error', 'Avg test error', 'Location','northwest')
title('1(b) : Average cross-validation Error - Poly deg 3');
ylabel('Average Cross-validation error');
xlabel('C');
saveas(H, '../plot/1b_pdeg3.png', 'png')

pdeg3_train_error
pdeg3_test_error

%% Plot section rbf

figure(2)
H = mesh(1:5, 1:5, rbf_train_error,'FaceLighting','gouraud','LineWidth',1);
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});
set(gca,'Ytick',1:5,'YTickLabel', {'1/32', '1/4', '1', '4', '32'});
xlabel('C')
ylabel('Sigma')
zlabel('Average Cross-validation error')
title('1(b) : Average train cross-validation Error - RBF');
saveas(H, '../plot/1b_rbf_train.png', 'png')

figure(1)
H = mesh(1:5, 1:5, rbf_test_error,'FaceLighting','gouraud','LineWidth',1);
set(gca,'Xtick',1:5,'XTickLabel', {'1', '10', '10^2', '10^3', '10^4'});
set(gca,'Ytick',1:5,'YTickLabel', {'1/32', '1/4', '1', '4', '32'});
xlabel('C')
ylabel('Sigma')
zlabel('Average Cross-validation error')
title('1(b) : Average test cross-validation Error - RBF');
saveas(H, '../plot/1b_rbf_test.png', 'png')



rbf_train_error
rbf_test_error

%% Testing with best param:
C = 10;
model = SVM_learner(train_data, train_labels, C, 'linear', 0);

y_pred = SVM_classifier(train_data, model);
linear_best_train = classification_error(train_labels, y_pred)

y_pred = SVM_classifier(test_data, model);
linear_best_test= classification_error(test_labels, y_pred)
decision_boundary_SVM(test_data, test_labels, model)


%% Best degree 2
C = 1;
model = SVM_learner(train_data, train_labels, C, 'poly', 2);

y_pred = SVM_classifier(train_data, model);
best_train = classification_error(train_labels, y_pred)

y_pred = SVM_classifier(test_data, model);
best_test = classification_error(test_labels, y_pred)
decision_boundary_SVM(test_data, test_labels, model)

%% Best degree 3
C = 100;
model = SVM_learner(train_data, train_labels, C, 'poly', 3);

y_pred = SVM_classifier(train_data, model);
best_train = classification_error(train_labels, y_pred)

y_pred = SVM_classifier(test_data, model);
best_test = classification_error(test_labels, y_pred)
decision_boundary_SVM(test_data, test_labels, model)

%% Best degree 3
C = 10;
sig = 1/4;
model = SVM_learner(train_data, train_labels, C, 'rbf', sig);

y_pred = SVM_classifier(train_data, model);
best_train = classification_error(train_labels, y_pred)

y_pred = SVM_classifier(test_data, model);
best_test = classification_error(test_labels, y_pred)
decision_boundary_SVM(test_data, test_labels, model)


