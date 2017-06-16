function log_jointprob = scoresToJointProb(pred_scores, preds)

ps = sigmoid(pred_scores);
correct_ps = abs(~preds - ps);
log_jointprob = sum(log(correct_ps));

end