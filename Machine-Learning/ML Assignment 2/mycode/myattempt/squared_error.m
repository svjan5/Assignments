function [err] = squared_error(y_pred,y_true)
% This function computes the average squared error between the predicted
% value vector and the actual value vector.
% y_pred - predicted real value vector of size  (n*1)
% y_true - actual real value vector of size (n*1)

%Output
%err - mean squared error between y_pred and y_true.

err = mean((y_pred-y_true).^2);
end

