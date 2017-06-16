function train_mat = cvToTrainMat(cv)
train_mat = false(cv.NumObservations, cv.NumTestSets);

for i = 1:cv.NumTestSets
    train_mat(cv.training(i), i) = true;
end

end