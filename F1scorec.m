function [F1] = F1score(gold, test_outcome)
% Where the input 'gold' refers to the gold standard classification
% 2 classes: 0/1
% 1 -->  p-val < THRESHOLD
% 'test_outcome' is the predicted classification 
% sens, spec, ppv and npv stand for sensitivity, specificity,
% positive predictive value, and negative predictive value

% True positives
TP = test_outcome(gold == 1);
TP = length(TP(TP == 1));

% False positives
FP = test_outcome(gold == 0);
FP = length(FP(FP == 1));

% False negatives
FN = test_outcome(gold == 1);
FN = length(FN(FN == 0));

% True negatives
TN = test_outcome(gold == 0);
TN = length(TN(TN == 0));

% Sensitivity
sens = TP/(TP+FN);

% Specificity
% spec = TN/(FP+TN);

% Positive predicted value
ppv = TP/(TP+FP);

% Negative predicted value
% npv = TN/(FN+TN);

% Accuracy
% acc = (TP+TN)/(TP+TN+FP+FN);

% False positive rate
% FPR = FP / (FP + TN);

% F1 score
F1 = 2* (ppv*sens)/(ppv+sens);