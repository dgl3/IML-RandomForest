load('example_dataset_1.mat');
parameter = 1;
lambda = 1;
K = compute_gram_model(data, parameter);
K2 = L2_distance(data,data,0);
maxK = max(max(K));
minK = min(min(K));
%Eigenvalues
eigenvalues = eig(K);
imagesc(K);
imagesc(K2);


[ model, other_values ] = train_dual_kernel_SVM_lambda(data,labels,lambda,K);
plot_dataset( data, labels, model );

[ model2, other_values2 ] = train_dual_kernel_SVM_lambda(data,labels,lambda,K2);
plot_dataset( data, labels, model2 );