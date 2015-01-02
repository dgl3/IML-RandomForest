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
costTest = test(tree,'test',X,Y);
costResubstitution = test(tree,'resubstitution');

%{
%Compute the full tree
fullTree = classregtree(X,Y,'prune','off');
view(fullTree);
fullTree
costTestFull = test(fullTree,'test',X,Y);
costResubstitutionFull = test(fullTree,'resubstitution');
%}