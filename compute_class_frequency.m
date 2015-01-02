function [ class1,class2 ] = compute_class_frequency( labels )
    class1 = sum(labels == 1);
    class2 = sum(labels == -1);
end

