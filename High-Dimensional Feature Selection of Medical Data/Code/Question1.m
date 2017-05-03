%% Read dataset
x=csvread('train_data1.csv',0,0,[0 0 699 128]);
y=csvread('train_data1.csv',0,129,[0 129 699 140]);
testX = csvread('test_feature_data1.csv',0,1);
number = csvread('test_feature_data1.csv',0,0,[0 0 203 0]);

%% Preprocessing
y(y(:,:)==-1) = 0;
net = patternnet(15);

net.trainFcn = 'trainrp';
net.trainParam.max_fail = 20;
net.trainParam.epochs = 1000;

net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;

results = zeros(204,13);
results(:,1)=number;
errors = zeros(1,700);
    
%% Divide into 12 sub problems
for i=1:12
    
    % T Test feature selection
    nowY = y(:,i);
    X0 = x(nowY==0,:);
    X1 = x(nowY==1,:);
    [~,p,~,~] = ttest2(X0,X1,'Vartype','unequal');
    [~,featureIdxSortbyP]= sort(p);
    nowX = x(:,featureIdxSortbyP(1:70));
    
    % LASSO regression feature selection
    [B,S] = lassoglm(nowX,nowY,'binomial','DFmax',30,'CV',10,'Alpha',0.5);
    model = B(:,S.IndexMinDeviance)~=0;
    nowX = nowX(:,model);

    nowTestX = testX(:,featureIdxSortbyP(1:70));
    nowTestX = nowTestX(:,model);
    
    % Train and use NN array
    outputs = zeros(1,204);
    for j=1:5
        rand = randperm(700);
        nowX = nowX(rand,:);
        nowY = nowY(rand,:);
        nowNet = train(net,nowX',nowY');
        errors = errors+gsubtract(nowY',nowNet(nowX'));
        outputs = outputs+nowNet(nowTestX');
    end
    
    % Generate result
    results(:,i+1) = outputs'/5;
end

%% Output result
disp(sum(abs(errors))/(700*60));
csvwrite('result1.csv',results);