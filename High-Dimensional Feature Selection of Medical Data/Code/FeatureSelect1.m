%% Read dataset and preprocessing
x=csvread('train_data1.csv',0,0,[0 0 699 128]);
y=csvread('train_data1.csv',0,129,[0 129 699 140]);
y(y(:,:)==-1) = 0;

%% T test
pSum = zeros(1,129);
for i=1:12
    nowY = y(:,i);
    X0 = x(nowY==0,:);
    X1 = x(nowY==1,:);
    [~,p,~,~] = ttest2(X0,X1,'Vartype','unequal');
    pSum = pSum+p;
end

[~,featureIdxSortbyP]= sort(pSum);
x=x(:,featureIdxSortbyP(1:50));

%% LASSO regression feature selection
opts = statset('UseParallel',true);
featureSum = zeros(12,1);
featureWeight = zeros(50,1);
for i=1:12
    [B,S] = lassoglm(x,y(:,i),'binomial','DFmax',30,'CV',10,'Alpha',0.5,'Options',opts);
    featureSum(i,1) = sum(B(:,S.IndexMinDeviance)~=0);
    featureWeight = featureWeight+B(:,S.IndexMinDeviance);
end

%% Filter
sum=floor(sum(featureSum)/12);
[~,featureIdxSortbyLasso]= sort(featureWeight);
x=x(:,featureIdxSortbyLasso(1:sum));
