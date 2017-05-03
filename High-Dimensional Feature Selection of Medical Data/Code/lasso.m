%% Read dataset
x=csvread('train_data2.csv',0,0,[0 0 3999 409]);
y=csvread('train_data2.csv',0,410,[0 410 3999 410]);

%% LASSO regression feature selection 

opts = statset('UseParallel',true);
% Linear Regression
[B,S] = lasso(x,y,'DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
% Poisson Regression
[B,S] = lassoglm(x,y,'poisson','DFmax',100,'CV',10,'Alpha',0.5,'Options',opts);
model = B(:,S.Index1SE)~=0;
