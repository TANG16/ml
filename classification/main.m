
function main
    trainAcc = zeros(5, 3);
    testAcc = zeros(5, 3);
    
    train = load('ionosphere_test.mat');
    test = load('ionosphere_train.mat');
    [trainAcc(1,:), testAcc(1,:)] = runCase(train, test);
    
    train = load('isolet_test.mat');
    test = load('isolet_train.mat');
    [trainAcc(2,:), testAcc(2,:)] = runCase(train, test);
    
    train = load('liver_test.mat');
    test = load('liver_train.mat');
    [trainAcc(3,:), testAcc(3,:)] = runCase(train, test);
    
    train = load('mnist_train.mat');
    test = load('mnist_test.mat');
    [trainAcc(4,:), testAcc(4,:)] = runCase(train, test);
    
    train = load('mushroom_test.mat');
    test = load('mushroom_train.mat');
    [trainAcc(5,:), testAcc(5,:)] = runCase(train, test);
    
    dd = 1:5;
    trainsvmLinear = [0.9160, 1, 0.585, 0.9875, 0.9998];
    testsvmLinear = [0.8713, 0.9833, 0.5724, 0.96, 0.9983];
    trainsvmRBF = [0.9760, 1, 0.665, 0.99, 1];
    testsvmRBF = [0.9505, 0.9917, 0.6276, 0.9675, 0.9993];
    
    figure(1);
	plot(dd,trainAcc(:,1),'b-*',dd,trainAcc(:,2),'r-*',dd,trainAcc(:,3),'k-*',dd,trainsvmLinear,'g-*',dd,testsvmLinear,'m-*', 'LineWidth', 2);
	legend('Logistic','kNN','kNN after PCA','SVM linear kernel','SVM RBF kernel','Location','best');
	xlabel('Data Set','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Training Data', 'fontsize', 14);
	grid on;
	figure(2);
	plot(dd,testAcc(:,1),'b-*',dd,testAcc(:,2),'r-*',dd,testAcc(:,3),'k-*',dd,trainsvmRBF,'g-*',dd,testsvmRBF,'m-*', 'LineWidth', 2);
	legend('Logistic','kNN','kNN after PCA','SVM linear kernel','SVM RBF kernel','Location','best');
	xlabel('Data Set','fontsize',12);
	ylabel('Acurracy','fontsize',12);
	title('Testing Data', 'fontsize', 14);
	grid on;
end

function [trainAcc, testAcc] = runCase(train, test)
    trainAcc = zeros(1, 3);
    testAcc = zeros(1, 3);
    
    t1 = cputime;
    [trainAcc(1), testAcc(1)] = Logistic(train.X, train.Y, test.X, test.Y);
    t2 = cputime;

    [k, trainAcc(2), testAcc(2), dim, trainAcc(3), testAcc(3)] = KNN(train.X, train.Y, test.X, test.Y);
    
    logiTi_k_dim = zeros(1,3);
    logiTi_k_dim(1) = t2 - t1; % logiii
    logiTi_k_dim(2) = k;
    logiTi_k_dim(3) = dim;    % pca dim
    
    logiTi_k_dim
    trainAcc
    testAcc
end