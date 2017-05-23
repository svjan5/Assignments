function [train_error, test_error] = getFoldError(C, X, y, X_test, y_test, kernelType, r)
    train_error = zeros(length(C_list), 1);
    test_error = zeros(length(C_list), 1);

    for i = 1:length(C_list)
        C = C_list(i);

        for k = 1:5
            X = train_data(kfold.training(k), :);
            y = train_labels(kfold.training(k));

            X_test = train_data(kfold.test(k), :);
            y_test = train_labels(kfold.test(k));

            model = SVM_learner(X, y, C, 'poly', 3);

            y_pred = SVM_classifier(X, model);
            train_error(i) = train_error(i) + classification_error(y_pred, y);

            y_pred = SVM_classifier(X_test, model);
            test_error(i) = test_error(i) + classification_error(y_pred, y_test);

        end
    end

    train_error = train_error / 5;
    test_error = test_error / 5;
end