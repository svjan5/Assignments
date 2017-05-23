epsilon = 0.1;
load('../data/regression_folds.mat');

C_list = [0.01;1;100];
test_error = zeros(length(C_list), 5);
train_error = zeros(length(C_list), 5);
tol = 0.01;

for i = 1:length(C_list)
    C = C_list(i);
    
    for k = 1:5
        train_data(:,:) = fold_train(k,:,:);
        train_labels(:, 1) = fold_train_y(k,:);
        test_data(:, :) = fold_test(k,:,:);
        test_labels(:,1) = fold_test_y(k,:);
        
        SVR_learner(train_data, train_labels, C, epsilon, tol, 'poly', 3);
        
        y_pred = kernel_predictor(train_data, 'svr_model');
        train_error(i,k) = squared_error(y_pred, train_labels);
        
        y_pred = kernel_predictor(test_data, 'svr_model');
        test_error(i,k) = squared_error(y_pred, test_labels);
    end
    C
end

[sound_y, sound_Fs] = audioread('/home/shikhar/Music/alert_signal.wav');
sound(sound_y, sound_Fs);
%%
test_error
train_error
% test_error = test_error / 5
% train_error = train_error / 5

avg_test_error = sum(test_error,2)/5
avg_train_error = sum(train_error,2)/5

H = plot(1:3, avg_train_error, 1:3, avg_test_error);
set(gca,'Xtick',1:3,'XTickLabel', {'0.01', '1', '100'});
legend('Avg train error', 'Avg test error', 'Location','northeast')
title('3(c) : Average cross-validation Error');
ylabel('Average Cross-validation error');
xlabel('lambda');

%% Testing with standard library
clc;

load('../data/regression_dataset'); test_y = test_y';
full_train_error = zeros(length(C_list), 1);
full_test_error = zeros(length(C_list), 1);

svr_model = fitrsvm(train,train_y,'KernelFunction','linear');

for i = 1:length(C_list)
    C = C_list(i);
    y_pred = svr_model.predict(train);
    full_train_error(i) = squared_error(y_pred, train_y);

    y_pred = svr_model.predict(test);
    full_test_error(i) = squared_error(y_pred, test_y);
end

full_test_error
full_train_error

%% Run on entire data set

epsilon = 0.1;


load('../data/regression_dataset'); test_y = test_y';
C_list = [0.01;1;100];
tol_list = 0:0.0011:0.01;

full_train_error = zeros(length(C_list), length(tol_list));
full_test_error = zeros(length(C_list), length(tol_list));


% for k = 1:length(tol_list)
%     tol = tol_list(k);
tol = 0.01;
    for i = 1:length(C_list)
        C = C_list(i);

        SVR_learner(train, train_y, C, epsilon, tol, 'poly', 3);

        y_pred = kernel_predictor(train, 'svr_model');
        full_train_error(i,k) = squared_error(y_pred, train_y);

        y_pred = kernel_predictor(test, 'svr_model');
        full_test_error(i,k) = squared_error(y_pred, test_y);
    end
% end


full_train_error
full_test_error

[sound_y, sound_Fs] = audioread('/home/shikhar/Music/alert_signal.wav');
sound(sound_y, sound_Fs);

