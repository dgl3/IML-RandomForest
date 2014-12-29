load('example_dataset_1.mat');
parameter = 1;
lambda = 1;
K = compute_gram_model(data, parameter);
maxK = max(max(K));
minK = min(min(K));
imagesc(K);
[ model, other_values ] = train_dual_kernel_SVM_lambda(data,labels,lambda,K);
plot_dataset( data, labels, model );