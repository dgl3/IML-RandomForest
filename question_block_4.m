load('example_dataset_1.mat');
K = 5;
indexes = create_KFolds(K,data,labels);
for i=1:size(indexes,1)
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels);
    %Kernel SVM
    %Average error surface
    
    %Tree- minparent
    %Average error surface
    tree = classregtree(X,Y);
    
end