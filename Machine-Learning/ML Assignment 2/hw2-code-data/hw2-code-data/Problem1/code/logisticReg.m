function y_pred = logisticReg(X_train, Y_train, X_test)


[m, d] = size(X_train);
X_train(:, d+1) = ones(m,1);
Y_train(Y_train == -1) = 0;

m_test = size(X_test, 1);
X_test(:, d+1) = ones(m_test,1);

w = zeros(d+1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 20000, 'MaxFunEvals', 100000);


[w, ~] = fminunc(@(t)(errorFunction(t, X_train, Y_train)), w, options);

% Predict on Testing data
y_pred  = zeros(m_test, 1);
for i = 1:1:m_test
    if (w'*X_test(i,:)' > 0)
        y_pred(i) = 1;
    else
        y_pred(i) = 0;
    end
end

y_pred(y_pred == 0) = -1;