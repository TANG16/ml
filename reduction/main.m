
function main
	load('10kTrain.mat');
	trainData = full(fea);
	trainLabel = gnd + 1;
	load('Test.mat');
	testData = full(fea);
	testLabel = gnd + 1;
	RunCase(trainData, trainLabel, testData, testLabel, 9, 20, 'MNIST_result.mat');

	load('COIL20.mat');
	indexDiff = setdiff(1:1440, 1:6:1440);
	trainData = fea(1:6:1440,:);
	trainLabel = gnd(1:6:1440,:);	
	testData = fea(indexDiff,:);
	testLabel = gnd(indexDiff,:);
	RunCase(trainData, trainLabel, testData, testLabel, 11, 40, 'COIL20_result.mat');
end


function RunCase(trainData, trainLabel, testData, testLabel, k1, k2, filename)

	%	1 	Raw data
	[RAWtrainAccu, RAWtestAccu] = BayesClassifier(trainData, trainLabel, testData, testLabel);

	save(filename, 'RAWtrainAccu', 'RAWtestAccu');

	%	2 	PCA
	dim = [1:k1];

	PCAtrainAccu = zeros(k1, 1);
	PCAtestAccu = zeros(k1, 1);
	for i = 1:k1
		[trainProjData, testProjData] = PCA(trainData, testData, dim(i));
		[PCAtrainAccu(i), PCAtestAccu(i)] = BayesClassifier(trainProjData, trainLabel, testProjData, testLabel);
	end

	save(filename, 'PCAtrainAccu', 'PCAtestAccu', '-append');

	figure(k1);
	plot(dim, PCAtrainAccu, 'k-*', 'LineWidth', 2);
	xlabel('Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Training Data with PCA', 'fontsize', 14);
	grid on;
	figure(k1 + 1);
	plot(dim, PCAtestAccu, 'k-*', 'LineWidth', 2);
	xlabel('Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Testing Data with PCA', 'fontsize', 14);
	grid on;

	%	3 	RLDA
	alpha = [0.0001, 0.001, 0.01, 0.1, 1];
	len = size(alpha,2);
	RLDAtrainAccu = zeros(k1, len);
	RLDAtestAccu = zeros(k1, len);
	for i = 1:k1
		for j = 1:len
			[trainProjData, testProjData] = RLDA(trainData, trainLabel, testData, dim(i), alpha(j));
			[RLDAtrainAccu(i,j), RLDAtestAccu(i,j)] = BayesClassifier(trainProjData, trainLabel, testProjData, testLabel);
		end
	end

	save(filename, 'RLDAtrainAccu', 'RLDAtestAccu', '-append');

	figure(k1*2);
	plot(dim,RLDAtrainAccu(:,1),'b-*',dim,RLDAtrainAccu(:,2),'r-*',dim,RLDAtrainAccu(:,3),'g-*',dim,RLDAtrainAccu(:,4),'m-*',dim,RLDAtrainAccu(:,5),'k-*', 'LineWidth', 2);
	legend('alpha = 0.0001','alpha = 0.001','alpha = 0.01','alpha = 0.1','alpha = 1','Location','best');
	xlabel('Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Training Data with RLDA', 'fontsize', 14);
	grid on;
	figure(k1*2 + 1);
	plot(dim,RLDAtestAccu(:,1),'b-*',dim,RLDAtestAccu(:,2),'r-*',dim,RLDAtestAccu(:,3),'g-*',dim,RLDAtestAccu(:,4),'m-*',dim,RLDAtestAccu(:,5),'k-*', 'LineWidth', 2);
	legend('alpha = 0.0001','alpha = 0.001','alpha = 0.01','alpha = 0.1','alpha = 1','Location','best');
	xlabel('Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Testing Data with RLDA', 'fontsize', 14);
	grid on;

	%	4 	PCA+LDA
	PCA_dim = [k2:20:100];
	len = size(PCA_dim,2);
	PCALDAtrainAccu = zeros(k1, len);
	PCALDAtestAccu = zeros(k1, len);
	for i = 1:k1
		for j = 1:len
			[trainProjData, testProjData] = PCALDA(trainData, trainLabel, testData, PCA_dim(j), dim(i));
			[PCALDAtrainAccu(i,j), PCALDAtestAccu(i,j)] = BayesClassifier(trainProjData, trainLabel, testProjData, testLabel);
		end
	end

	save(filename, 'PCALDAtrainAccu', 'PCALDAtestAccu', '-append');

	figure(k1*3);
	if len == 5
		plot(dim,PCALDAtrainAccu(:,1),'b-*',dim,PCALDAtrainAccu(:,2),'r-*',dim,PCALDAtrainAccu(:,3),'g-*',dim,PCALDAtrainAccu(:,4),'m-*',dim,PCALDAtrainAccu(:,5),'k-*', 'LineWidth', 2);
		legend('PCA d = 20','PCA d = 40','PCA d = 60','PCA d = 80','PCA d = 100', 'Location','best');
	else
		plot(dim,PCALDAtrainAccu(:,1),'b-*',dim,PCALDAtrainAccu(:,2),'r-*',dim,PCALDAtrainAccu(:,3),'m-*',dim,PCALDAtrainAccu(:,4),'k-*', 'LineWidth', 2);
		legend('PCA d = 40','PCA d = 60','PCA d = 80','PCA d = 100','Location','best');
	end
	xlabel('LDA Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Training Data with PCA+LDA', 'fontsize', 14);
	grid on;
	figure(k1*3 + 1);
	if len == 5
		plot(dim,PCALDAtestAccu(:,1),'b-*',dim,PCALDAtestAccu(:,2),'r-*',dim,PCALDAtestAccu(:,3),'g-*',dim,PCALDAtestAccu(:,4),'m-*',dim,PCALDAtestAccu(:,5),'k-*', 'LineWidth', 2);
		legend('PCA d = 20','PCA d = 40','PCA d = 60','PCA d = 80','PCA d = 100','Location','best');
	else
		plot(dim,PCALDAtestAccu(:,1),'b-*',dim,PCALDAtestAccu(:,2),'r-*',dim,PCALDAtestAccu(:,3),'m-*',dim,PCALDAtestAccu(:,4),'k-*', 'LineWidth', 2);
		legend('PCA d = 40','PCA d = 60','PCA d = 80','PCA d = 100','Location','best');
	end
	xlabel('LDA Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Testing Data with PCA+LDA', 'fontsize', 14);
	grid on;


	%	compare different methods
	figure(k1*4);
	plot(dim,PCAtrainAccu,'b-*',dim,RLDAtrainAccu(:,5),'r-*',dim,PCALDAtrainAccu(:,1),'k-*', 'LineWidth', 2);
	legend('PCA','RLDA alpha = 1','PCA+LDA d = 40','Location','best');
	xlabel('Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Training Data', 'fontsize', 14);
	grid on;
	figure(k1*4 + 1);
	plot(dim,PCAtestAccu,'b-*',dim,RLDAtestAccu(:,5),'r-*',dim,PCALDAtestAccu(:,1),'k-*', 'LineWidth', 2);
	legend('PCA','RLDA alpha = 1','PCA+LDA d = 40','Location','best');
	xlabel('Dimension','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Testing Data', 'fontsize', 14);
	grid on;

end
