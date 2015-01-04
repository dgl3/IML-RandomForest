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

%aus.mat
dataAus = SAus.Data;
labelsAus = SAus.labels;

perm = randperm(size(dataAus,2));
dataAus = dataAus(:,perm);
labelsAus = labelsAus(perm);

K = 10;
indexes = create_KFolds(K,dataAus,labelsAus);
EoutAus = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),dataAus,labelsAus');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutAus = [EoutAus; Eout];
end

%bcw.mat
data = SBcw.BreastWisc.dat;
labels = SBcw.BreastWisc.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutBcw = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutBcw = [EoutBcw; Eout];
end

%bid.mat
data = SBid.Bupa.dat;
labels = SBid.Bupa.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutBid = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutBid = [EoutBid; Eout];
end


%bre.mat
data = SBre.Breast.dat;
labels = SBre.Breast.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutBre = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutBre = [EoutBre; Eout];
end

%car.mat
data = SCar.Data.dat;
labels = SCar.Data.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutCar = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutCar = [EoutCar; Eout];
end

%cmc.mat
data = SCmc.Data.dat;
labels = SCmc.Data.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutCmc = [];
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
    EoutCmc = [EoutCmc; Eout];
end

%ech.mat
data = SEch.Data.dat;
labels = SEch.Data.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutEch = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutEch = [EoutEch; Eout];
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
end


%ger.mat
data = SGer.German.dat;
labels = SGer.German.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutGer = [];
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
    EoutGer = [EoutGer; Eout];
end

%hec.mat
data = SHec.HeartClev.dat;
labels = SHec.HeartClev.labels;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

K = 10;
indexes = create_KFolds(K,data,labels);
EoutHec = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),data,labels');
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',3);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted) == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutHec = [EoutHec; Eout];
end
