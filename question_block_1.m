load('example_dataset_1.mat');
sigma = 1;
lambda = 1;
K = compute_gram_model(data, data, sigma);
maxK = max(K(:));
minK = min(K(:));
%Eigenvalues
eigenvalues = eig(K);
imagesc(K);

model = train_dual_kernel_SVM_lambda(data,labels,lambda,K);
plot_dataset( data, labels, model, sigma );