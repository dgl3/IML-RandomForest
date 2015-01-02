function [ ] = plot_dataset( data, labels, model, K )
    figure;
    if labels(1) == 1
        scatter(data(1,1),data(2,1),5,'red');
    else
        scatter(data(1,1),data(2,1),5,'green');
    end
    hold on;
    for i=2:size(data,2)
        if labels(i) == 1
            scatter(data(1,i),data(2,i),5,'red');
        else
            scatter(data(1,i),data(2,i),5,'green');
        end
    end
    xlabel('x1');
    ylabel('x2');
    lineX=[];
    lineY=[];
    %lineYpos=[];
    %lineYneg=[];
    for i=1:size(data,2)
        X = data(1,i);
        lineX=[lineX, X];
        Y = getY(i,model,K,labels);
        lineY=[lineY, Y];
        %lineYpos=[lineYpos, Ypos];
        %lineYneg=[lineYneg, Yneg];
    end
    plot(lineX, lineY);
    %plot(lineX, lineYpos, 'g--');
    %plot(lineX, lineYneg, 'g--');
    hold off;
end

