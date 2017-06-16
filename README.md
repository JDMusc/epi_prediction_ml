# Epilepsy Prediction Machine Learning

This code performs the Bayesian-based machine learning pipeline described in the paper “__Using machine learning to classify Temporal Lobe Epilepsy based on diffusion MRI__”.

_Note: this code requires a Matlab installation with the [Statistics and Machine Learning Toolbox](https://www.mathworks.com/help/stats/fitcsvm.html)_

The code can be used for any binary-classification problem with a small enough dataset that can be loaded locally into memory. The steps are below:

1. load a matrix (X) of size n by p.
	* n is the number of subjects/observations
	* p is the number of features.
2. load a vector of binary classifications (y) of size n by 1.
3. Execute “runAllIters”.
	* \>\> tbl = runAllIters(X, y, 10);
	* the experiment ran for 1000 iterations, but 10 may be enough for a quick check
4. To view the results of a particular run, execute:
	* \>\> run\_index = 1;
	* \>\> calculateMetricsPreds(tbl.pred(tbl.run == run\_index), y)
