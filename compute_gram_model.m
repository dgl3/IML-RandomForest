function [ K ] = compute_gram_model( data, sigma )
    dist = L2_distance(data,data,0);
    K = exp(-dist/(2*sigma.^2));
end

