
%	trainProjData:	projected data of training data set after PCALDA
%	testProjData:	projected data of testing data set after PCALDA
%	trainData:	input traning data in m-by-d matrix, m data row number, d input dimension
%	trainLabel:	class label of training data in m-by-1 matrix
%	testData:	input testing data
%	dim1:	user-specified dimension for PCA
%	dim2:	user-specified dimension for LDA

function [trainProjData, testProjData] = PCALDA(trainData, trainLabel, testData, dim1, dim2)

	[trainProjData, testProjData] = PCA(trainData, testData, dim1);
	[trainProjData, testProjData] = LDA(trainProjData, trainLabel, testProjData, dim2);
end

function [trainProjData, testProjData] = LDA(trainData, trainLabel, testData, dim)

	[trainProjData, testProjData] = RLDA(trainData, trainLabel, testData, dim, 0);
end