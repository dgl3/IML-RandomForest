function Y = getY(xi,model,K,labels)
    Y = sum(K(:,xi).*(labels.*model));
end

