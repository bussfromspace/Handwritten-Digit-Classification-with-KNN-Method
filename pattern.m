%% PATTERN RECOGNITION ASSIGNMENT
clear;
characters = load ('characters2.mat');
Chars1Stroke = characters.Chars1Stroke;
Chars2Stroke = characters.Chars2Stroke;
Labels1Stroke = characters.Labels1StrokeChars;
Labels2Stroke = characters.Labels2StrokeChars;

Chars = cat(3,Chars1Stroke,Chars1Stroke);
Chars = cat(1, Chars, Chars2Stroke);
Labels = cat(2,Labels1Stroke,Labels2Stroke);


% testing = Centralize(reshape(Chars1Stroke(2,:,:),[size(Chars1Stroke,2),size(Chars1Stroke,3)])');
% figure;
%  subplot(1,2,1);
%  scatter(Chars1Stroke(2,1,:),Chars1Stroke(2,2,:)), title('1Stroke Digit Before Centralization');
%  subplot(1,2,2);scatter(testing(:,1),testing(:,2));title('1Stroke Digit After Centralization');
%  
%  testing2 = Centralize(reshape(Chars2Stroke(4,:,:),[size(Chars2Stroke,2),size(Chars2Stroke,3)])');
%  figure;
%  subplot(1,2,1);
%  scatter(Chars2Stroke(4,1,:),Chars2Stroke(4,2,:)), title('2Stroke Digit Before Centralization');
%  subplot(1,2,2);scatter(testing2(:,1),testing2(:,2));title('2Stroke Digit After Centralization');
%  
%  
%  testing = NormalizeSize(testing);
%  figure;
%  %subplot(1,2,1);
%  %scatter(Chars1Stroke(2,1,:),Chars1Stroke(2,2,:)), title('1Stroke Digit Before Normalization');
%  %subplot(1,2,2);
%  scatter(testing(:,1),testing(:,2));title('1Stroke Digit After Normalization');
%  
%  testing2 = NormalizeSize(testing2);
%  figure;
%  %subplot(1,2,1);
%  %scatter(Chars2Stroke(4,1,:),Chars2Stroke(4,2,:)), title('2Stroke Digit Before Normalization');
%  %subplot(1,2,2);
%  scatter(testing2(:,1),testing2(:,2));title('2Stroke Digit After Normalization');
 
 

% Charsx = reshape(Chars(:,1,:),[size(Chars,1),size(Chars,3)]);
% Charsy = reshape(Chars(:,2,:),[size(Chars,1),size(Chars,3)]);
% [zx,Scorex] = my_pca(Charsx,2);
% [zy,Scorey] = my_pca(Charsy,2);
% figure;scatter(zx(:,1),zx(:,2));
% figure;scatter(zy(:,1),zy(:,2));

k = 30; %used for k-fold selection
d = 25; %used to reduce the number of data points per stroke
n = 50; % number of data points per stroke
d_max = 5e2; % max evaluation of the distance in DTW 
KNN = 7; %number of k nearest neighbours
N = size(Chars,1);
Indices = crossvalind('Kfold',N,k);
total_accuracy = 0;
f_score = 0;
predict_vec = 0;
test_vec = 0;

for i = 1:k
    %select partitions for each for cycle
    trainingLabel = Labels(:,(Indices ~= i));
    testLabel = Labels(:,(Indices == i));
    trainingData = Chars((Indices ~= i),:,:);
    testData = Chars((Indices == i),:,:);
    
    tic;
    
%     trainingData = Decimate(trainingData,size(trainingData,3)/n,d);
%     testData     = Decimate(testData,size(testData,3)/n,d);
    for j = 1:size(trainingData,1)
        trainingData_new2(j,:,:) = NormalizeSize(Centralize(reshape(trainingData(j,:,:),[size(trainingData,2),size(trainingData,3)])'));
        %training(j,:,:) = Centralize(reshape(training(j,:,:),[size(training,2),size(training,3)]));
    end
    for j = 1:size(testData,1)
        testData_new2(j,:,:) = NormalizeSize(Centralize(reshape(testData(j,:,:),[size(testData,2),size(testData,3)])'));
        %test(j,:,:) = Centralize(reshape(test(j,:,:),[size(test,2),size(test,3)]));
    end
    
    trainingData_new = Decimate(trainingData_new2,size(trainingData_new2,2)/n,d);
    testData_new     = Decimate(testData_new2,size(testData_new2,2)/n,d);
    
     accuracy = 0;  
     
%     for j = 1:size(testData,1)
%         test = reshape(testData_new(j,:,:),[size(testData_new,2),size(testData_new,3)]);
%         dist = zeros(1, size(trainingData,1));
%         %prediction = zeros(1, size(testData,1));
%         for jj = 1:size(trainingData,1)
%             training = reshape(trainingData_new(jj,:,:),[size(trainingData_new,2),size(trainingData_new,3)]);
%             [dist(jj),distM] = DtwDistance(test,training, d_max);
%         end
%         [dist,idx] = sort(dist, 2, 'ascend');
%         %K nearest neighbors
%         dist = dist(:,1:KNN);
%         idx = idx(:,1:KNN);
%         %majority vote
%         prediction(j) = mode(trainingLabel(idx),2);
%        % X = ['prediction for ',num2str(testLabel(j)),'is: ', num2str(prediction(j))];
%        %disp(X)
        prediction = knn_func( trainingData_new, trainingLabel,testData_new, KNN,d_max);
        accuracy = accuracy + (prediction == testLabel);
        %accuracy = accuracy + (prediction(j) == testLabel(j));
   % end    
    timer(i) = toc; 
    total_accuracy = total_accuracy + (accuracy)/size(testData,1);
    f_score = f_score + f1score(prediction,testLabel);
    predict_vec = cat(2,predict_vec, prediction);
    test_vec = cat(2,test_vec, testLabel);
end
total_accuracy = sum(total_accuracy) / k;
f_score = f_score/k;
display(['accuracy: %',num2str(total_accuracy)]);
display(['mean f1 score: ',num2str(f_score)]);
display(['mean elapsed time: ', num2str(mean(timer)),' seconds']);

[F1, Confusion] = f1score(predict_vec(1,2:end),test_vec(1,2:end));
imshow(Confusion, [], 'InitialMagnification', 1600);
colorbar;
axis on;
% Set up figure properties:
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0 1 1]);
% Get rid of tool bar and pulldown menus that are along top of figure.
set(gcf, 'Toolbar', 'none', 'Menu', 'none');
% Give a name to the title bar.
set(gcf, 'Name', 'Demo by ImageAnalyst', 'NumberTitle', 'Off')
%plotconfusion(test_vec(1,2:end),predict_vec(1,2:end));
%colormap(gray);





