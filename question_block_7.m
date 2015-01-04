addpath('datasets');
SAus= load('aus.mat');
SBcw = load('bcw.mat');
SBid = load('bid.mat');
SBre = load('bre.mat');
SCar = load('car.mat');
SCmc = load('cmc.mat');
SEch = load('ech.mat');
SFac = load('fac.mat');
SGer = load('ger.mat');
SHec = load('hec.mat');

sigma=0.1;
lambda=0.1;

%aus.mat
dataAus = SAus.Data;
labelsAus = SAus.labels;

perm = randperm(size(dataAus,2));
dataAus = dataAus(:,perm);
labelsAus = labelsAus(perm);

K = 10;
indexes = create_KFolds(K,dataAus,labelsAus);
EoutAusDT = [];
EoutAusSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),dataAus,labelsAus');
     %Tree
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutAusDT = [EoutAusDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutAusSVM = [EoutAusSVM; errorTest];
end

%bcw.mat
data = SBcw.BreastWisc.dat;
labels = SBcw.BreastWisc.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutBcwDT = [];
EoutBcwSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutBcwDT = [EoutBcwDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutBcwSVM = [EoutBcwSVM; errorTest];
end

%bid.mat
data = SBid.Bupa.dat;
labels = SBid.Bupa.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutBidDT = [];
EoutBidSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutBidDT = [EoutBidDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutBidSVM = [EoutBidSVM; errorTest];
end


%bre.mat
data = SBre.Breast.dat;
labels = SBre.Breast.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutBreDT = [];
EoutBreSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutBreDT = [EoutBreDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutBreSVM = [EoutBreSVM; errorTest];
end

%car.mat
data = SCar.Data.dat;
labels = SCar.Data.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutCarDT = [];
EoutCarSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutCarDT = [EoutCarDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutCarSVM = [EoutCarSVM; errorTest];
end

%cmc.mat
data = SCmc.Data.dat;
labels = SCmc.Data.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutCmcDT = [];
EoutCmcSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    yPredicted = str2num(cell2mat(yPredicted));
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutCmcDT = [EoutCmcDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutCmcSVM = [EoutCmcSVM; errorTest];
end

%ech.mat
data = SEch.Data.dat;
labels = SEch.Data.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutEchDT = [];
EoutEchSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutEchDT = [EoutEchDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutEchSVM = [EoutEchSVM; errorTest];
end

%fac.mat
data = SFac.faces.dat;
labels = SFac.faces.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutFac = [];
EoutFacSVM =[];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutFac = [EoutFac; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutFacSVM = [EoutFacSVM; errorTest];
end


%ger.mat
data = SGer.German.dat;
labels = SGer.German.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutGerDT = [];
EoutGerSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    yPredicted = str2num(cell2mat(yPredicted));

    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutGerDT = [EoutGerDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutGerSVM = [EoutGerSVM; errorTest];
end

%hec.mat
data = SHec.HeartClev.dat;
labels = SHec.HeartClev.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutHecDT = [];
EoutHecSVM = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutHecDT = [EoutHecDT; Eout];
    
    %SVM
    KTrain = compute_gram_model(XTrain, XTrain, sigma);
    model = train_dual_kernel_SVM_lambda(XTrain, YTrain, lambda, KTrain);
    KTest = compute_gram_model(XTrain, XTest, sigma);
    errorTest = test_dual_kernel_SVM_lambda(YTest, YTrain, model, KTest);
    EoutHecSVM = [EoutHecSVM; errorTest];
end
