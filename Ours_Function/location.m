function [value,loc] = location(mx)
    [m,m1] = size(mx);
    [value,index] = min(mx(:));
    col = ceil(index / m);
    row = index - (col - 1) * m;
    loc = [row, col];
end