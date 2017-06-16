function s = calculateMetrics(corrects, y, positive_val, negative_val)

if ~exist('positive_val', 'var')
    positive_val = 1;
end

if ~exist('negative_val', 'var')
    negative_val = 0;
end

ps = y == positive_val;
ns = y == negative_val;

tps = corrects & ps;
fps = ~corrects & ns;

tns = corrects & ns;
fns = ~corrects & ps;

tp_counts = sum(tps, 1);
fp_counts = sum(fps, 1);

tn_counts = sum(tns, 1);
fn_counts = sum(fns, 1);

precision = tp_counts ./ (tp_counts + fp_counts);
recall = tp_counts ./ (tp_counts + fn_counts);

f1 = 2 * precision .* recall ./ (precision + recall);

s.f1 = f1;
s.accuracy = sum(corrects, 1)./ length(y);
s.tpr = recall;
s.tnr = tn_counts ./ (tn_counts + fp_counts);
s.precision = precision;
s.recall = recall;
s.sensitivity = s.tpr;
s.specificity = s.tnr;


end
