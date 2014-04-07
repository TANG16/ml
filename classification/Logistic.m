
function [trainAccuracy, testAccuracy] = Logistic(trainData, trainLabel, testData, testLabel)
    % run n times, return average accuracy
    times = 10;
    trainAccuracy = 0;
    testAccuracy = 0;
    for i = 1:times 
        [t1, t2] = run(trainData, trainLabel, testData, testLabel);
        trainAccuracy = trainAccuracy + t1;
        testAccuracy = testAccuracy + t2;
    end
    
    trainAccuracy = trainAccuracy / times;
    testAccuracy = testAccuracy / times;
end

function [trainAccuracy, testAccuracy] = run(trainData, trainLabel, testData, testLabel)
    w = Train(trainData, trainLabel);
    trainAccuracy = Test(w, trainData, trainLabel);
    testAccuracy = Test(w, testData, testLabel);
end

function w = Train(trainData, trainLabel)
	[dataNum, dim] = size(trainData);
    
    % [-0.01, 0.01]
    w = rand(dim + 1, 1) * 0.02 - 0.01;
    
    stepSize = 0.3;
    threshold = 0.02;
    iterMax = 2000;
    
    diff = 1;
    iter = 0;
    while diff > threshold && iter <= iterMax
        % o = wx   n*(d+ 1) (d+1)*1
        o = [ones(dataNum,1), trainData] * w;   
        % y = sigmoid(o)  n * 1
        y = 1 ./ (1 + exp(-o));                 
        
        % dw = sumisum( (r-y)x )   (d+1)*n n*1
        deltaW = [ones(dataNum,1), trainData]' * (trainLabel - y); 
        w = w + stepSize * deltaW;
        
        diff = sum(abs(deltaW));
        iter = iter + 1;
    end
end

function accuracy = Test(w, testData, testLabel)
    dataNum = size(testLabel, 1);
    y = [ones(dataNum,1), testData] * w;
    accuracy = sum(testLabel == (y > 0.5)) / dataNum;
end