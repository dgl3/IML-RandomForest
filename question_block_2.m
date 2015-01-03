%Consider that you train a full decision tree. What is the expected training error
%to obtain? Why?
%The expected training error is 0.
load('example_dataset_1.mat');
X = data';
Y = labels;
%Compute a tree with the default parameters
tree = classregtree(X,Y);
view(tree)
tree

yPredicted = eval(tree,X);
incorrectPredictions = sum((Y == yPredicted) == 0);
trainingError = (incorrectPredictions/size(data,2))*100;

notPrunedTree = classregtree(X,Y, 'prune', 'off', 'minparent',1);
view(notPrunedTree)
notPrunedTree

yPredicted2 = eval(notPrunedTree,X);
incorrectPredictions2 = sum((Y == yPredicted2) == 0);
trainingError2 = (incorrectPredictions2/size(data,2))*100;