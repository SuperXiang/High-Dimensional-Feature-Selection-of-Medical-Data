%% Read dataset and preprocessing
x=csvread('train_data2.csv',0,0,[0 0 3999 409]);
y=csvread('train_data2.csv',0,410,[0 410 3999 410]);
X1 = x(y==0,:);
X2 = x(y==1,:);
X3 = x(y==2,:);

%% T test
[~,p12,~,~] = ttest2(X1,X2,'Vartype','unequal');
[~,p13,~,~] = ttest2(X2,X3,'Vartype','unequal');
[~,p23,~,~] = ttest2(X1,X3,'Vartype','unequal');
p=p12+p13+p23;
[~,featureIdxSortbyP]= sort(p);

%% LASSO regression feature selection

x=x(:,featureIdxSortbyP(1:100));
opts = statset('UseParallel',true);
% Linear Regression
[B,S] = lasso(x,y,'DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
% Poisson Regression
% [B,S] = lassoglm(x,y,'poisson','DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
model = B(:,S.Index1SE)~=0;
x=x(:,model);

%% SVM
