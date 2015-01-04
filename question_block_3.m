%K-Fold crossvalidation
load('example_dataset_1.mat');
K = 10;

%randomly sorting the data
perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

indexes = create_KFolds(K,data,labels);
[class1,class2] = compute_class_frequency(labels);
frequency = class1/size(data,2);
class1TrainFreq = [];
class2TrainFreq = [];

class1TestFreq = [];
class2TestFreq = [];

for i=1:size(indexes,1)
   [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels);
   %Compute class frequencey for each K-Fold
   [trainClass1, trainClass2] = compute_class_frequency(YTrain);
   class1TrainFreq = [class1TrainFreq;trainClass1/size(YTrain,1)];
   class2TrainFreq = [class2TrainFreq;trainClass2/size(YTrain,1)];

   [testClass1, testClass2] = compute_class_frequency(YTest);
   class1TestFreq = [class1TestFreq;testClass1/size(YTest,1)];
   class2TestFreq = [class2TestFreq;testClass2/size(YTest,1)];

end