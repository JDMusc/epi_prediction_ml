function [preds, scores] = runIter(X, y, train_mat, show_plots, save_states, state_dir)

if ~exist('train_mat', 'var')
    cv = cvpartition(y, 'kFold', 5);
    train_mat = cvToTrainMat(cv);
end

if ~exist('show_plots', 'var')
    show_plots = false;
end

if ~exist('save_states', 'var')
    save_states = false;
end

if ~exist('state_dir', 'var')
    state_dir = 'state_info';
end

num_subjs = length(y);

preds = zeros(num_subjs, 1);
scores = preds;

kernel_fn = 'linear';

num_folds = size(train_mat, 2);

function saveData(~)
    var_name = inputname(1);
    save([state_dir '/' var_name '.mat'], var_name);
end

num_cs = 19;
num_features = size(X, 2);
coefs = zeros(num_features, 1);
if save_states
   mkdir(state_dir);
   cs_mat = zeros(num_cs, num_folds);
   ps_mat = cs_mat;
   bias = zeros(num_cs, 1);
   preds_mat = zeros(num_subjs, num_cs);
end

for i = 1:num_folds
    train_ixs = train_mat(:, i);
    
    X_train = X(train_ixs, :);
    y_train = y(train_ixs);
    
    test_ixs = ~train_ixs;
    
    X_test = X(test_ixs, :);
    
    fitC = @(xt, yt, c) fitClassifier(xt, yt, c, kernel_fn);
    mn_c = findCrange(X_train, y_train, fitC);
    max_c = mn_c * 5;
    cs = linspace(mn_c, max_c, 20)';
    disp([mn_c max_c]);
    logp_d_st_c = cs;
    parfor i_c = 1:length(cs)
        c = cs(i_c);
        class_i = fitC(X_train, y_train, c);
        [~, scores_i] = class_i.predict(X_train);
        logp_d_st_c(i_c) = scoresToJointProb(scores_i(:, 2), y_train);
    end

    elbow_ix = findElbow(logp_d_st_c);
    max_c = cs(elbow_ix);

    if show_plots
        figure(1);
        plot(cs, logp_d_st_c, '*');
        hold on; 
        plot(max_c, logp_d_st_c(elbow_ix), 'o', 'MarkerFaceColor', 'red');
        hold off;
        xlabel('C');
        ylabel('Log Likelihood_D_|_C');
        title('Log Likelihood_D_|_C');
        pause(.1);
    end

    std_c = (max_c - mn_c)/2;
    cs = [linspace(max(mn_c - max_c, exp(-7)), mn_c, 10) linspace(mn_c, max_c, 10)]';
    cs = [cs(1:10); cs(12:end)];

    p_c = normpdf(cs, mn_c, std_c);
    p_c = p_c ./ sum(p_c);

    logp_d_st_c = cs;
    parfor i_c = 1:length(cs)
        c = cs(i_c);
        class_i = fitC(X_train, y_train, c);
        [~, scores_i] = class_i.predict(X_train);
        logp_d_st_c(i_c) = scoresToJointProb(scores_i(:, 2), y_train);
    end

    p_d_st_c = exp(logp_d_st_c);
    p_d = sum(p_d_st_c .* p_c);
    p_c_st_d = (p_d_st_c .* p_c) / p_d;

    if show_plots
        figure(2);
        subplot(3, 1, 1); plot(cs, p_c, '*'); title('P_C (prior)');
        subplot(3, 1, 2); plot(cs, p_d_st_c, '*'); title('Likelihood_D_|_C');
        subplot(3, 1, 3); plot(cs, p_c_st_d, '*'); title('P_C_|_D (posterior)');
        xlabel('C');
        pause(.1);
    end

    preds_mat_curr = zeros(sum(test_ixs), length(cs));
    coefs_curr = zeros(num_features, 1);
    bias_curr = zeros(num_cs, 1);
    parfor i_c = 1:length(cs)
        c = cs(i_c);
        class_i = fitC(X_train, y_train, c);
        [~, scores_i] = class_i.predict(X_test);
        preds_mat_curr(:, i_c) = sigmoid(scores_i(:, 2));
        coefs_curr = coefs_curr + class_i.Beta .* p_c_st_d(i_c);
        bias_curr(i_c) = bias_curr(i_c) + class_i.Bias * p_c_st_d(i_c);
    end

    scores_curr = preds_mat_curr * p_c_st_d;

    preds(test_ixs) = scores_curr > .5;
    scores(test_ixs) = scores_curr;

    if save_states
        cs_mat(:, i) = cs;
        ps_mat(:, i) = p_c_st_d;
        preds_mat(test_ixs, :) = preds_mat_curr;
        coefs = coefs + 1/num_folds .* coefs_curr;
        bias = bias + 1/num_folds .* bias_curr;
    end
end

if save_states
    saveData(cs_mat);
    saveData(ps_mat);
    saveData(preds);
    saveData(scores);
    saveData(preds_mat);
    saveData(coefs);
    saveData(bias);
    saveData(train_mat);
end

end
