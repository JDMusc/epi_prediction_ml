function s = calculateMetricsPreds(preds, y, positive_val, negative_val)

if ~exist('positive_val', 'var')
    positive_val = 1;
end

if ~exist('negative_val', 'var')
    negative_val = 0;
end

s = calculateMetrics(preds == y, y, positive_val, negative_val);

end
