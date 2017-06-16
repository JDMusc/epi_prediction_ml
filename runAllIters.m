function tbl = runAllIters(X, y, num_runs, show_plots, save_states)

num_obs = length(y);
num_folds = 5;

if ~exist('num_runs', 'var')
    num_runs = 1000;
end

if ~exist('show_plots', 'var')
    show_plots = false;
end

if ~exist('save_states', 'var')
    save_states = false;
end

train_mat_all = false(num_obs, num_folds, num_runs);

function tbl = createTable()
    tbl = table();
    tbl.pred = zeros(num_obs * num_runs, 1);
    tbl.subj = repmat((1:num_obs)', num_runs, 1);
    
    runs = repmat(1:num_runs, num_obs, 1);
    tbl.run = runs(:);
end

tbl = createTable();

    function predAndSave(X, run)
        fprintf('predict %d\n', run);
        
        dir_name = sprintf('%d', run);
        
        train_mat = train_mat_all(:, :, run);
        preds = runIter(X, y, train_mat, show_plots, save_states, dir_name);
        
        ixs = tbl.run == run;
        tbl.pred(ixs) = preds;
    end

for i = 1:num_runs
    disp(i);
    cv = cvpartition(y, 'kFold', num_folds); 
    train_mat_all(:, :, i) = cvToTrainMat(cv);
    
    predAndSave(X, i);
end

writetable(tbl);

end
