
function [trainProjData, testProjData, dim] = PCA(trainData, testData)

    [EigVector, dim] = chooseDim(trainData);
    
	trainProjData = PCAproj(trainData, EigVector);
	testProjData = PCAproj(testData, EigVector);
end

function [EigVector, dim] = chooseDim(trainData)
	covar = cov(trainData);
	originDim = size(trainData, 2);
    
    [originEigVector, EigValue] = eigs(covar, originDim - 1);
    
    eigValue = diag(EigValue);
    total = sum(eigValue);
    
    PoV = 0;
    dim = 0;
    counter = 0;
    while PoV <= 0.9
        dim = dim + 1;
        counter = counter + eigValue(dim);
        PoV = counter / total;
    end
    
    EigVector = originEigVector(:,1:dim);
end

function ProjectedData = PCAproj(data, EigVector)
	[n, d] = size(data);
	M = mean(data);
	subtractMean = data - repmat(M, n, 1);
	ProjectedData = subtractMean * EigVector;
end