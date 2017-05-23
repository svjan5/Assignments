load('../data/regression_dataset.mat');
load('../data/regression_folds.mat');

CArr = [0.01; 1; 100];
errFoldTestArr = [];
errFoldTrainArr = [];
foldArr=[];
meanErrFoldTestArr = [];
meanErrFoldTrainArr = [];
CFoldArr = [];
minErr = inf;
optC = 0.01;
r=1;
kerneltype='linear';
ebsilon = 0.1;

%%
for l=1:3
    C = CArr(l)
    sumErrTrain = 0;
    sumErrTest = 0;
    
    for f=1:5
        f
        sizeTrain = size(fold_train,2);
        model=SVR_learner(squeeze(fold_train(f,:,:)),squeeze(fold_train_y(f,:))', C, kerneltype, r, ebsilon);
        y_pred = zeros(sizeTrain,1);
        for i=1:sizeTrain
            x = squeeze(fold_train(f,i,:));
            y_pred(i) = kernel_predictor(model,x);
        end
        trainFoldErr = squared_error(y_pred, squeeze(fold_train_y(f,:))')
        sumErrTrain = sumErrTrain + trainFoldErr;
        
        sizeTest = size(fold_test,2);
        y_pred = zeros(sizeTest,1);
        for i=1:sizeTest
            x = squeeze(fold_test(f,i,:));
            y_pred(i) = kernel_predictor(model,x);
        end
        testFoldErr = squared_error(y_pred,squeeze( fold_test_y(f,:))')
        sumErrTest = sumErrTest + testFoldErr;
        
        errFoldTestArr = [errFoldTestArr ;testFoldErr];
        errFoldTrainArr = [errFoldTrainArr  ;trainFoldErr];
        CFoldArr = [CFoldArr; C];
        foldArr = [foldArr ;f];
    end
    if minErr > sumErrTest/5
        minErr = sumErrTest/5;
        optC = C;
    end
    meanErrFoldTestArr = [meanErrFoldTestArr; sumErrTest/5];
    meanErrFoldTrainArr = [meanErrFoldTrainArr; sumErrTrain/5];
end

%%
errTestArr = [];
errTrainArr = [];

for l=1:3
    C = CArr(l);
    sumErrTrain = 0;
    sumErrTest = 0;
   
        sizeTrain = size(train,1);
        model=SVR_learner(train,train_y, C, kerneltype, r, ebsilon);
       
        
        y_pred = zeros(sizeTrain,1);
        for i=1:sizeTrain
            x = train(i,:)';
            y_pred(i) = kernel_predictor(model,x);
        end
        trainErr = squared_error(y_pred, train_y)
        
        sizeTest = size(test,1);
        y_pred = zeros(sizeTest,1);
        for i=1:sizeTest
             x = test(i,:)';
            y_pred(i) = kernel_predictor(model,x);
        end
        testErr = squared_error(y_pred, test_y')
    
    errTestArr = [errTestArr; testErr];
    errTrainArr = [errTrainArr; trainErr];
end

%% print tables

T1 = table(CFoldArr, foldArr, errFoldTrainArr, errFoldTestArr, 'VariableNames', {'C','FoldNumber','TrainingError','TestError'});
disp(T1);

T2 = table(CArr, meanErrFoldTrainArr, meanErrFoldTestArr, 'VariableNames', {'C','AverageTrainingError','AverageTestError'});
disp(T2);

T3 = table(optC, minErr, 'VariableNames', {'OptC','MinAverageTestError'});
disp(T3);

T4 = table(CArr, errTrainArr, errTestArr, 'VariableNames', {'C','FullTrainingError','FullTestError'});
disp(T4);
