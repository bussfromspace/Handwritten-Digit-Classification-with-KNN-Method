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


k = 10; %used for k-fold selection
d = 25; %used to reduce the number of data points per stroke
n = 50; % number of data points per stroke
d_max = 5e2; % max evaluation of the distance in DTW 
KNN = 1; %number of k nearest neighbours
N = size(Chars,1);
Indices = crossvalind('Kfold',N,k);
total_accuracy = 0;
f_score = 0;

for i = 1:k
    %select partitions for each for cycle
    trainingLabel = Labels(:,(Indices ~= i));
    testLabel = Labels(:,(Indices == i));
    trainingData = Chars((Indices ~= i),:,:);
    testData = Chars((Indices == i),:,:);
    
    tic;
    
    trainingData = Decimate(trainingData,size(trainingData,3)/n,d);
    testData     = Decimate(testData,size(testData,3)/n,d);
    for j = 1:size(trainingData,1)
        trainingData_new(j,:,:) = Centralize(NormalizeSize(reshape(trainingData(j,:,:),[size(trainingData,2),size(trainingData,3)])'));
        %training(j,:,:) = Centralize(reshape(training(j,:,:),[size(training,2),size(training,3)]));
    end
    for j = 1:size(testData,1)
        testData_new(j,:,:) = Centralize(NormalizeSize(reshape(testData(j,:,:),[size(testData,2),size(testData,3)])'));
        %test(j,:,:) = Centralize(reshape(test(j,:,:),[size(test,2),size(test,3)]));
    end
    
     accuracy = 0;  
     
    for j = 1:size(testData,1)
        test = reshape(testData_new(j,:,:),[size(testData_new,2),size(testData_new,3)]);
        dist = zeros(1, size(trainingData,1));
        %prediction = zeros(1, size(testData,1));
        for jj = 1:size(trainingData,1)
            training = reshape(trainingData_new(jj,:,:),[size(trainingData_new,2),size(trainingData_new,3)]);
            [dist(jj),distM] = DtwDistance(test,training, d_max);
        end
        [dist,idx] = sort(dist, 2, 'ascend');
        %K nearest neighbors
        dist = dist(:,1:KNN);
        idx = idx(:,1:KNN);
        %majority vote
        prediction(j) = mode(trainingLabel(idx),2);
       % X = ['prediction for ',num2str(testLabel(j)),'is: ', num2str(prediction(j))];
       %disp(X)
        accuracy = accuracy + (prediction(j) == testLabel(j));
    end    
    timer(i) = toc; 
    total_accuracy = total_accuracy + (accuracy)/size(testData,1);
    f_score = f_score + f1score(prediction,testLabel);
end
    total_accuracy = total_accuracy / k;
    f_score = f_score/k;
    display(['accuracy: %',num2str(total_accuracy)]);
    display(['mean f1 score: ',num2str(f_score)]);
    display(['mean elapsed time: ', num2str(mean(timer)),' seconds']);
    




