load('example_dataset_1.mat');
K = 5;

perm = randperm(size(data,2));
data = data(:,perm);
labels = labels(perm);

%External Cross-Validation
indexes = create_KFolds(K,data,labels);
Eout = [];
for i=1:K
    [XTrainOut,YTrainOut,XTestOut,YTestOut] = get_partitions(indexes(i,:),data,labels);
    %Nested Cross-Validation
    indexes = create_KFolds(K,XTrainOut,YTrainOut);
    
    trainingErrorFold1 = [];
    testErrorFold1 = [];

    trainingErrorFold2 = [];
    testErrorFold2 = [];

    trainingErrorFold3 = [];
    testErrorFold3 = [];

    trainingErrorFold4 = [];
    testErrorFold4 = [];

    trainingErrorFold5 = [];
    testErrorFold5 = [];

    trainingErrorFold10 = [];
    testErrorFold10 = [];

    trainingErrorFold15 = [];
    testErrorFold15 = [];
    for j=1:K
        [XTrain,YTrain,XTest,YTest] = get_partitions(indexes(i,:),XTrainOut,YTrainOut);
        %Tree - minparent 1
        tree1 = classregtree(XTrain', YTrain, 'minparent',1);
        %view(tree1);
        tree1;

        yPredicted = eval(tree1,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError1 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold1 = [trainingErrorFold1; trainingError1];


        yPredicted = eval(tree1,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError1 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold1 = [testErrorFold1; testError1];


        %Tree - minparent 2
        tree2 = classregtree(XTrain', YTrain, 'minparent',2);
        %view(tree2);
        tree2;

        yPredicted = eval(tree2,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError2 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold2 = [trainingErrorFold2; trainingError2];


        yPredicted = eval(tree2,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError2 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold2 = [testErrorFold2; testError2];


        %Tree - minparent 3
        tree3 = classregtree(XTrain', YTrain, 'minparent',3);
        %view(tree3);
        tree3;

        yPredicted = eval(tree3,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError3 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold3 = [trainingErrorFold3; trainingError3];


        yPredicted = eval(tree3,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError3 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold3 = [testErrorFold3; testError3];


        %Tree - minparent 4
        tree4 = classregtree(XTrain', YTrain, 'minparent',4);
        %view(tree4);
        tree4;

        yPredicted = eval(tree4,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError4 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold4 = [trainingErrorFold4; trainingError4];


        yPredicted = eval(tree4,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError4 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold4 = [testErrorFold4; testError4];

        %Tree - minparent 5
        tree5 = classregtree(XTrain', YTrain, 'minparent',5);
        %view(tree5);
        tree5;

        yPredicted = eval(tree5,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError5 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold5 = [trainingErrorFold5; trainingError5];


        yPredicted = eval(tree5,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError5 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold5 = [testErrorFold5; testError5];

        %Tree - minparent 10
        tree6 = classregtree(XTrain', YTrain, 'minparent',10);
        %view(tree6);
        tree6;

        yPredicted = eval(tree6,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError6 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold10 = [trainingErrorFold10; trainingError6];


        yPredicted = eval(tree6,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError6 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold10 = [testErrorFold10; testError6];

        %Tree - minparent 15
        tree7 = classregtree(XTrain', YTrain, 'minparent',10);
        %view(tree6);
        tree7;

        yPredicted = eval(tree7,XTrain');
        incorrectPredictions = sum((YTrain == yPredicted) == 0);
        trainingError7 = (incorrectPredictions/size(XTrain,2))*100;
        trainingErrorFold15 = [trainingErrorFold15; trainingError7];


        yPredicted = eval(tree7,XTest');
        incorrectPredictions = sum((YTest == yPredicted) == 0);
        testError7 = (incorrectPredictions/size(XTest,2))*100;
        testErrorFold15 = [testErrorFold15; testError7];
        
        
    end
    %Minparent = 1, 2 or 3 -> I will use minparent equal to 1
    tree = classregtree(XTrainOut', YTrainOut, 'minparent',1);
    yPredicted= eval(tree,XTestOut');
    
    incorrectPredictions = sum((YTestOut == yPredicted) == 0);
    error = (incorrectPredictions/size(XTestOut,2))*100;
    Eout = [Eout; error];

end
