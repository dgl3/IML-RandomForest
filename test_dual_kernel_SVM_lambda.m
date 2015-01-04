function [ error ] = test_dual_kernel_SVM_lambda( labelsTest, labelsTrain, model, K )
    Y = getY(model,K,labelsTrain);
    errorI = sign(Y.*labelsTest);
    error = sum(abs(Y(errorI==-1)));
end

