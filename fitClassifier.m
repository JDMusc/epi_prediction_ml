function csvm = fitClassifier(X, y, c, kernel_fn)

if ~exist('kernel_fn', 'var')
    kernel_fn = 'linear';
end
        
csvm = fitcsvm(X, y, ...
    'KernelScale', 'auto', ...
    'KernelFunction', kernel_fn, ...
    'BoxConstraint', c, ...
    'Standardize', true, ...
    'ClassNames', [false; true]);

end