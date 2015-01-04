function [  XTrain, YTrain ] = get_fold( index, data, labels )
    start = index(1);
    finish = index(2);
    %XTest and YTest partitions
    XTrain = data(:,start:finish);
    YTrain = labels(start:finish);


end

