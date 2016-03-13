function [F1] = F1score_multiclass(All_gold, All_test_outcome)
% Where the input 'gold' refers to the gold standard classification
% 2 classes: 0/1
% 1 -->  p-val < THRESHOLD
% 'test_outcome' is the predicted classification 
% sens, spec, ppv and npv stand for sensitivity, specificity,
% positive predictive value, and negative predictive value

num_classes = max(All_gold);
F1s = zeros(num_classes,1);
weights = zeros(num_classes,1); 
for i=1:num_classes
    
    gold=All_gold;
    test_outcome=All_test_outcome;
    for j=1:size(All_gold,1)
        if gold(j)==i
            gold(j)= 1;
        else
            gold(j)= 0;
        end
        if test_outcome(j)==i
            test_outcome(j)= 1;
        else
            test_outcome(j)= 0;
        end       
    end
    
    
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
    F1s(i) = 2* (ppv*sens)/(ppv+sens);
    weights(i) = sum(gold);
end

for i=num_classes:-1:1
    if isnan(F1s(i))
        F1s(i) = 0;
    end
    F1s(i) = F1s(i) * weights(i);
end

F1 = sum(F1s)/sum(weights);

end