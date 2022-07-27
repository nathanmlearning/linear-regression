file = 'aumentados_day_features.xlsx'
opts = detectImportOptions(file);
opts.SelectedVariableNames = [1:13];
aumentadosdayfeatures = readmatrix(file, opts)

clear opts file 

Train = []
Test = []
for i = [1, 2, 3, 4]
    SubM = aumentadosdayfeatures(aumentadosdayfeatures(:,13)==i,:);
    [m,n] = size(SubM) ;
    P = 700 ;
    idx = randperm(m)  ;
    Training = SubM(idx(1:1700),:) ; 
    Testing = SubM(idx(1700+1:end),:) ;
    Train = [Train;Training];
    Test = [Test; Testing]
end



clear Testing Training idx P SubM m n i aumentadosdayfeatures

xTrainPrev = Train(:,1:12)

xTrainPrev(xTrainPrev == 0) = 0.000001;

xTrainPrev = round(xTrainPrev, 0);
subplot(1,2,1);
plot(xTrainPrev(1,:));

title("ariginal");


yTrain = Train(:,13)
xTrain = normalize(xTrainPrev,2);

subplot(1,2,2);
plot(xTrain(1,:));
title("Normalizada");

xTestPrev = Test(:,1:12)
xTestPrev(xTestPrev == 0) = 0.000001;
xTestPrev = round(xTestPrev, 0);
xTest = normalize(xTestPrev,2);
yTest = Test(:,13)

% Y = tsne(xTrain);
% colors = ['r', 'b', 'g', 'm']
% sym = '.'
% siz = [16]
% gscatter(Y(:,1), Y(:,2), yTrain, colors,sym,siz)
% Y1 = [Y yTrain]
clear Test Train Y colors sym siz Y1

%correlacion
[rho, pval] = corr(xTrain);


numClass = unique(yTrain);
mOnehotEncondingTrain = (yTrain==1:length(numClass));

clear numClass

[yTestPredicted] = myNeuralNetworkFunction(xTest);
[yTrainPredicted] = myNeuralNetworkFunction(xTrain);

[~,idx] = max(yTrainPredicted');
errorModelTraining = sum(idx' ~= yTrain)/length(yTrain);
fprintf('error de entrenamiento>  %f2 \n', errorModelTraining)


[~,idx] = max(yTestPredicted');
errorModelTesting = sum(idx' ~= yTest)/length(yTest);
fprintf('error de testeo> %f2 \n', errorModelTesting)
