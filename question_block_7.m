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
K = 10;
indexes = create_KFolds(K,SAus.Data,SAus.labels);
EoutAus = [];
for i=1:K
    [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),SAus.Data,SAus.labels);
     %Tree - minparent 1
    tree = classregtree(XTrain', YTrain, 'minparent',1);
    %view(tree1);
    tree;
    
    yPredicted = eval(tree,XTest');
    incorrectPredictions = sum((YTest == yPredicted') == 0);
    Eout = (incorrectPredictions/size(XTest,2))*100;
    EoutAus = [EoutAus; Eout];
end

%bcw.mat

