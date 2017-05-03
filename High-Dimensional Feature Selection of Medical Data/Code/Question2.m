%% Read dataset and preprocessing
x=csvread('train_data2.csv',0,0,[0 0 3999 409]);
y=csvread('train_data2.csv',0,410,[0 410 3999 410]);
X1 = x(y==0,:);
X2 = x(y==1,:);
X3 = x(y==2,:);

%% T Test
[~,p12,~,~] = ttest2(X1,X2,'Vartype','unequal');
[~,p13,~,~] = ttest2(X2,X3,'Vartype','unequal');
[~,p23,~,~] = ttest2(X1,X3,'Vartype','unequal');
p=p12+p13+p23;
[~,featureIdxSortbyP]= sort(p);

%% LASSO regression feature selection

x=x(:,featureIdxSortbyP(1:100));
opts = statset('UseParallel',true);
% Linear Regression
[B,S] = lasso(x,y,'DFmax',50,'CV',10,'Alpha',0.5,'Options',opts);
% Poisson Regression
% [B,S] = lassoglm(x,y,'poisson','DFmax',50,'CV',10,'Alpha',0.5,'Options',opts);
model = B(:,S.Index1SE)~=0;
x=x(:,model);

clear X1 X2 X3 p p12 p13 p23 B S

%% Process class label
newY=zeros(4000,3);
newY(y==0,1)=1;
newY(y==1,2)=1;
newY(y==2,3)=1;
y=newY;
clear newY;

%% Build NN parameters
neurons = sum(model~=0);
net = patternnet(floor(neurons/2));
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

net.trainFcn = 'trainrp';
net.trainParam.max_fail = 20;
net.trainParam.epochs = 1000;

%% Train NN array
errorsArray=zeros(1,20);
nets = cell(1,20);

for i=1:20
    % Randomly disorganize sample order
    rand = randperm(4000);
    x = x(rand,:);
    y = y(rand,:);

    nets{i} = train(net,x',y');
    neti=nets{i};
    outputs = neti(x');
    errors = gsubtract(y',outputs);
    errorsArray(i) = sum(sum(abs(errors)))/12000;
end

%% Select 10 best NN by error information

[~,IdxSortbyerrors]= sort(errorsArray,'ascend');
nets = nets(IdxSortbyerrors(1:10));
disp(mean(errorsArray(IdxSortbyerrors(1:10))));
clear errors errorsArray i net neti neurons opts outputs rand IdxSortbyerrors

%% Read test data
testX = csvread('test_feature_data2.csv',0,1);
number = csvread('test_feature_data2.csv',0,0,[0 0 1030 0]);
testX = testX(:,featureIdxSortbyP(1:100));
testX = testX(:,model);

%% Classify by NN
testY = zeros(3,1031);
for i=1:10
    neti = nets{i};
    testY = testY+neti(testX');
end
testY = testY/10;
testY = (vec2ind(testY)-1)';
testY = [number,testY];
csvwrite('result2.csv',testY);
