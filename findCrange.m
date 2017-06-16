function c_range = findCrange(X, y, fitC, min_acc)

function diff = accDist(c, dest)
    if ~exist('dest', 'var')
        dest = 1;
    end
    
    csvm = fitC(X, y, c);

    acc = sum(csvm.predict(X) == y)/length(y);
    diff = dest - acc;
end

lower_max_c = 10^(-7);
max_c = 10^7;

function calc_c = calcC(min_c, max_c) 
    if max_c < 10 * min_c
        calc_c = (min_c + max_c)/2;
    else
        calc_c = 10^((log10(max_c) + log10(min_c))/2);
    end
end

while max_c - lower_max_c > .1
    c = calcC(lower_max_c, max_c);
    diff = accDist(c);
    if diff > 0
        lower_max_c = c;
    else
        max_c = c;
    end
end

if exist('min_acc', 'var')
    min_c = 10^(-7);
    min_c = fminbnd(@(c) accDist(c, min_acc), min_c, max_c);

    c_range = [min_c, max_c];
else
    c_range = max_c;
end

end