function [ K ] = compute_gram_model( data, parameter )
    numInstances = size(data,2);
    K = zeros(numInstances, numInstances, 'double');
    for i=1:numInstances
        xi = data(i);
        for j=1:numInstances
            xj = data(j);
            numerator = -L2_distance(xi,xj,0).^2;
            denominator = (2*parameter).^2;
            Kij = exp(numerator/denominator);
            K(i,j) = Kij;
        end
    end

end

