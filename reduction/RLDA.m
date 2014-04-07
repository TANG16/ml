
%	trainProjData:	projected data of training data set after RLDA
%	testProjData:	projected data of testing data set after RLDA
%	trainData:	input traning data in m-by-d matrix, m data row number, d input dimension
%	trainLabel:	class label of training data in m-by-1 matrix
%	testData:	input testing data
%	dim:	user-specified dimension for RLDA
%	alpha:	regularization parameter

function [trainProjData, testProjData] = RLDA(trainData, trainLabel, testData, dim, alpha)
	uniqueLabel = unique(trainLabel);
	classNum = size(uniqueLabel, 1);

	[m, d] = size(trainData);
	Sw = zeros(d);
	Sb = zeros(d);
	dataMean = mean(trainData);

	for i = 1:classNum
		index = find(trainLabel == uniqueLabel(i));

		sampleNum = size(index, 1);
		sampleData = trainData(index,:);
		sampleMean = mean(sampleData);

		Sw = Sw + (sampleNum - 1) * cov(sampleData);
		Sb = Sb + sampleNum * (sampleMean - dataMean)' * (sampleMean - dataMean);
	end

	S = pinv(Sw + alpha * eye(d)) * Sb;
	[EigVector, EigValue] = eigs(S, dim);

	trainProjData = LDAproj(trainData, EigVector);
	testProjData = LDAproj(testData, EigVector);
end

function ProjectedData = LDAproj(data, EigVector)
	ProjectedData = data * EigVector;
end