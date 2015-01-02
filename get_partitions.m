function [ XTrain, YTrain, XTest, YTest ] = get_partitions( index, data, labels)
    start = index(1);
    finish = index(2);
    %XTest and YTest partitions
    XTest = data(:,start:finish);
    YTest = labels(start:finish);
    
    %XTrain and YTrain partitions
    if start == 1
        XTrain = data(:,finish+1:size(data,2));
        YTrain = labels(finish+1:size(data,2));
    else if finish == size(data,2)
        XTrain = data(:,1:start-1);
        YTrain = labels(1:start-1);
        else
            XTrain1 = data(:,1:start-1);
            XTrain2 = data(:,finish+1:size(data,2));
            XTrain = [XTrain1, XTrain2];
            
            YTrain1 = labels(1:start-1);
            YTrain2 = labels(finish+1:size(data,2));
            YTrain = [YTrain1;YTrain2];
        end
    
    end

end

