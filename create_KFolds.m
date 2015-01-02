function [ testIndexes ] = create_KFolds( K, data, labels )
    testIndexes = zeros(K,2,'int16');
    sizeFold = int16(size(data,2)/10);
    testIndexes(1,1) = 1;
    testIndexes(1,2) = sizeFold;
    for i=1:K-1
        testIndexes(i,1) = (i-1)*sizeFold+1;
        testIndexes(i,2) = i*sizeFold;
    end
    testIndexes(K,1) = (K-1)*sizeFold+1;
    testIndexes(K,2) = size(data,2);
end

