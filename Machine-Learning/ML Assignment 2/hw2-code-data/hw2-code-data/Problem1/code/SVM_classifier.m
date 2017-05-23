function labels = SVM_classifier(testdata, model)
    % INPUT
    % testdata - m X n matrix of the test data samples
    % model    - SVM model structure returned by SVM_learner
    
    % OUTPUT
    % labels - m x 1 vector of predicted labels
    
    % Write code here
    [m,n] = size(testdata);
    labels = zeros(m, 1);
    supp_vecs = model.support_vectors;
    supp_y = model.support_vectors_y;
    
    
    for i = 1:m
        val = model.b;
        for j = 1:size(supp_vecs, 1)
            val = val + model.alphas(j) * supp_y(j) * compute_kernel(supp_vecs(j,:), testdata(i,:), model.kerneltype, model.r);
        end
        
        if(val >= 0)
            labels(i) = 1;
        else
            labels(i) = -1;
        end
    end
    
%     for i = 1:m
%         if (testdata(i,:)*model.w + model.b > 0)
%             labels(i) = 1;
%         else
%             labels(i) = -1;
%         end
%     end
end
