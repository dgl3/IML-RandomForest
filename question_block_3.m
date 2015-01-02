%K-Fold crossvalidation
load('example_dataset_1.mat');
indexes = create_KFolds(10,data,labels);
for i=1:size(indexes,1)
   [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels);
   %Create Tree and compute the error
   
end