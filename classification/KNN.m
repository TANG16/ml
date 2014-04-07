
function [k, trainAcc, testAcc, dim, PCAtrainAcc, PCAtestAcc] = KNN(trainData, trainLabel, testData, testLabel)
    t1 = cputime;
    % choose k from 1 to 10
    k = bestK(trainData, trainLabel);
    t2 = cputime;
    
    trainAcc = run(k, trainData, trainLabel, trainData, trainLabel);
    testAcc = run(k, trainData, trainLabel, testData, testLabel);
    t3 = cputime;
    
    % with pca
    [trainProjData, testProjData, dim] = PCA(trainData, testData);  
    PCAtrainAcc = run(k, trainProjData, trainLabel, trainProjData, trainLabel);
    PCAtestAcc = run(k, trainProjData, trainLabel, testProjData, testLabel);
    t4 = cputime;
    
    choK_knn_knnPCA = zeros(1,3);
    choK_knn_knnPCA(1) = t2 - t1; % choose k
    choK_knn_knnPCA(2) = t3 - t2; % knn time
    choK_knn_knnPCA(3) = t4 - t3; % knn after pca
    
    choK_knn_knnPCA
end

function k = bestK(trainData, trainLabel)
    dataNum = size(trainLabel,1);
    dataNum_80 = round(0.8 * dataNum);
    maxAccuary = 0;
    
    for n = 1:10
        randomOrder = randperm(dataNum);
        
        ttrainData = trainData(randomOrder, :);
        ttrainLabel = trainLabel(randomOrder, :);
        
        % 80% as train data
        D1 = ttrainData(1:dataNum_80,:);
        D2 = ttrainData(dataNum_80:end,:);
        
        % 20% as test data
        L1 = ttrainLabel(1:dataNum_80,:);
        L2 = ttrainLabel(dataNum_80:end,:);
    
        acc = run(n, D1, L1, D2, L2);
        
        if acc > maxAccuary
            maxAccuary = acc;
            k = n;
        end
    end
end

function accuracy = run(k, trainData, trainLabel, testData, testLabel)
    
    dataNum = size(testLabel,1);
    result = zeros(dataNum, 1);
    for n = 1:dataNum
        result(n) =  testPoint(k, trainData, trainLabel, testData(n, :));
    end
    accuracy = sum(testLabel == result) / dataNum;
end

function resultLabel = testPoint(k, trainData, trainLabel, point)
    % euclidean^2
    dataNum = size(trainLabel,1);
    distance = (trainData - repmat(point, dataNum, 1)).^2;
    distance = sum(distance, 2);
    
    [dummy, index] = sort(distance, 'ascend');
    sortedLabel = trainLabel(index);
    accurateNum = sum(sortedLabel(1:k));
    
    if accurateNum == k/2
        resultLabel = rand(1) > 0.5;
    else
        resultLabel = accurateNum > k/2;
    end    
end