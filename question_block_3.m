%K-Fold crossvalidation
load('example_dataset_1.mat');
K = 10;
indexes = create_KFolds(K,data,labels);
[class1,class2] = compute_class_frequency(labels);
for i=1:size(indexes,1)
   [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels);
   %Compute class frequencey for each K-Fold
   [trainClass1, trainClass2] = compute_class_frequency(YTrain);
   [testClass1, testClass2] = compute_class_frequency(YTest);
end