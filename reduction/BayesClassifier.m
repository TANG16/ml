
function [trainAccu, testAccu] = BayesClassifier(trainData, trainLabel, testData, testLabel)

	[classNum, priors, means, sigmas] = Training(trainData, trainLabel);

	trainAccu = Testing(trainData, trainLabel, classNum, priors, means, sigmas);
	testAccu = Testing(testData, testLabel, classNum, priors, means, sigmas);
end


function [classNum, priors, means, sigmas] = Training(trainData, trainLabel)

	uniqueLabel = unique(trainLabel);
	classNum = size(uniqueLabel, 1);

	[totalNum, d] = size(trainData);

	priors = zeros(classNum,1);
	means = zeros(classNum,d);
	sigmas = cell(classNum, 1);

	for i = 1:classNum
		index = find(trainLabel == uniqueLabel(i));
		SampleData = trainData(index,:);
		SampleNum = size(index, 1);

		means(i,:) = mean(SampleData);
		priors(i) = SampleNum / totalNum;
		sigmas{i} = cov(SampleData) * (SampleNum - 1) / SampleNum;
	end
end


function accuracy = Testing(data, label, classNum, priors, means, sigmas)

	[totalNum, d] = size(data);

	classProb = zeros(totalNum,classNum);
	Predictions = zeros(totalNum,1);

	for i = 1:classNum
		invSigma = pinv(sigmas{i});
		subtractMean = data - repmat(means(i,:), totalNum, 1);
		temp = subtractMean *  invSigma .* subtractMean * ones(d, 1);

		detSigma = abs(det(sigmas{i}));
		
		classProb(:,i) = - 0.5*log(detSigma) - 0.5 * temp + log(priors(i));
	end

	[dummy, Predictions] = max(classProb');
	accuracy = sum(Predictions' == label) / totalNum;
end