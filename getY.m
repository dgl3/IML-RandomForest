function [ Y, Ypos, Yneg ] = getY( X, w )
    Y = (X*w(2)+w(1))/(-1*w(3));
    Ypos = (X*w(2)+w(1)-1)/(-1*w(3));
    Yneg = (X*w(2)+w(1)+1)/(-1*w(3));
end

