clear; clc;
load('../data/regression_dataset.mat'); test_y = test_y';

SVR_learner(train, train_y, 0.1, 0.000001, 'linear', 0);

[sound_y, sound_Fs] = audioread('/home/shikhar/Music/alert_signal.wav');
sound(sound_y, sound_Fs);
%% Try model

y_pred = kernel_predictor(train, 'svr_model');
train_error = squared_error(y_pred, train_y)

y_pred = kernel_predictor(test, 'svr_model');
test_error = squared_error(y_pred, test_y)