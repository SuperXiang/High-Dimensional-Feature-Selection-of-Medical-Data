%% Read dataset

testX = csvread('test_feature_data2.csv',0,1);
number = csvread('test_feature_data2.csv',0,0,[0 0 1030 0]);
testX = testX(:,featureIdxSortbyP(1:100));
testX = testX(:,model);

%% Neural Network
testY = net(testX');
testY = (vec2ind(testY)-1)';
testY = [number,testY];

%% Outpur data
csvwrite('result2.csv',testY);