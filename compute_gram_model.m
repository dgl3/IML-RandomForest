function [ K ] = compute_gram_model( data1, data2, sigma )
    dist = L2_distance(data1,data2,0);
    K = exp(-dist/(2*sigma.^2));
end

