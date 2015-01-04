function Y = getY(model,K,labels)
    Y = K'*(labels.*model);
end

