function acc = Classification_Accuracy(y_pred, y_true)
% This function computes the classification accuracy for the predicted labels
% with respect to the ground truth. The returned accuracy value is a real
% number between 0, 100 [in percentage terms]

% y_true: vector of true labels (each label +1/-1)
% y_pred: vector of predicted labels (each prediction +1/-1)
% err: classification error (fraction of misclassifications)

	acc = (length(find(y_pred == y_true)) / length(y_true))*100;
end
