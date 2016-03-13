function [F1 C] = f1score( predicted, observed )
C = confusionmat(observed,predicted);
%precision
precision = (diag(C)+1) ./ (sum(C,2)+1);
precision = mean(precision);
%recall
recall = (diag(C)+1) ./ (sum(C,1)'+1);
recall = mean(recall);
F1 = 2* (precision*recall)/(precision + recall);
