function [prediction] = knn_func( trainingData, trainingLabel, testData,...
KNN, d_max)
    for j = 1:size(testData,1)
        test=reshape(testData(j,:,:),[size(testData,2),size(testData,3)]);
        dist=zeros(1, size(trainingData,1));
        for jj = 1:size(trainingData,1)
            training=reshape(trainingData(jj,:,:),[size(trainingData,2),...
            size(trainingData,3)]);
            %calculate the DTW distance and save it to dist array
            [dist(jj),distM] = DtwDistance(test,training, d_max);
        end
        %sort the distances in order
        [dist,idx] = sort(dist, 2, 'ascend');
        %K nearest neighbors with closest KNN indexes
        dist = dist(:,1:KNN);
        idx = idx(:,1:KNN);
        %majority vote
        prediction(j) = mode(trainingLabel(idx),2);
    end  
end

