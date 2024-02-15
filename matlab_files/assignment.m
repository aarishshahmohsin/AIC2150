filename = 'cancer.csv';
dataTable = readtable(filename);

disp('First few rows of the dataset:');
disp(head(dataTable));

features = dataTable(:, 2:end);
targetVariable = dataTable(:, 1);

rng(42);
splitRatio = 0.8;
idx = randperm(size(dataTable, 1));
trainingIdx = idx(1:round(splitRatio * length(idx)));
testingIdx = idx(round(splitRatio * length(idx)) + 1:end);

trainingData = dataTable(trainingIdx, :);
testingData = dataTable(testingIdx, :);

X_train = table2array(trainingData(:, 2:end));
y_train = table2array(trainingData(:, 1));

X_test = table2array(testingData(:, 2:end));
y_test = table2array(testingData(:, 1));

k = 3;
mdl = fitcknn(X_train, y_train, 'NumNeighbors', k);

y_test_categorical = categorical(y_test);
y_test_numeric = double(y_test_categorical);
confMat = confusionmat(y_test, predict(mdl, X_test));

TP = confMat(2, 2);
TN = confMat(1, 1);
FP = confMat(1, 2);
FN = confMat(2, 1);

accuracy = (TP + TN) / sum(confMat(:));

