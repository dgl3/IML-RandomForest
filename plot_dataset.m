function [ ] = plot_dataset( data, labels, model, sigma )
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
    
    d = 0.02;
    [x1Grid,x2Grid] = meshgrid(min(data(1,:)):d:max(data(1,:)),min(data(2,:)):d:max(data(2,:)));
    xGrid = [x1Grid(:),x2Grid(:)];
    
    KGrid = compute_gram_model(data, xGrid', sigma);
    YGrid = getY(model,KGrid,labels);
    
    contour(x1Grid,x2Grid,reshape(YGrid,84,87));
    
    hold off;
end

