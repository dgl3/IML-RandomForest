function [ model, other_values ] = train_dual_kernel_SVM_lambda( data, labels, lambda, K )
    m = size(data,2);
    
    %Q = diag(labels)*data'*data*diag(labels);
    %K(i,j) = K(xi,xj), where x is equal to data
    Q = diag(labels)*K*diag(labels);
    cvx_begin
        variables v(m);
        maximize( v'*ones(m,1) - 0.5*v'*Q*v )
        subject to
            0 <= v;
            v <= lambda;
            v'*labels == 0;
    cvx_end

    w = data*(v.*labels);
    b = -(w'*data-1/labels)*v/sum(v);
    model = [b;w];
    
    images = labels'.*(w'*data+b)-1;
    indNeg = images<=0;
    errorRate = sum(images(indNeg==1))*-1;
    
    other_values = struct('lambda',lambda,'errorRate',errorRate);
end

