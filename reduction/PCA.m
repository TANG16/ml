
%	trainProjData:	projected data of training data set after PCA
%	testProjData:	projected data of testing data set after PCA
%	trainData:	input traning data in m-by-d matrix, m data row number, d input dimension
%	testData:	input testing data
%	dim:	user-specified dimension for PCA

function [trainProjData, testProjData] = PCA(trainData, testData, dim)

	covar = cov(trainData);
	[EigVector, EigValue] = eigs(covar, dim);
	trainProjData = PCAproj(trainData, EigVector);
	testProjData = PCAproj(testData, EigVector);
end

function ProjectedData = PCAproj(data, EigVector)
	[n, d] = size(data);
	M = mean(data);
	subtractMean = data - repmat(M, n, 1);
	ProjectedData = subtractMean * EigVector;
end