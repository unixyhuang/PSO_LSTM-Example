function result = fitness(pop)

global inputn outputn shuru_num shuchu_num XValidation YValidation
tic

pop(1)=round(pop(1));
layers = [ ...
    sequenceInputLayer(shuru_num)
    lstmLayer(pop(1))
    fullyConnectedLayer(shuchu_num)
    regressionLayer];
options = trainingOptions('adam', ...  % 梯度下降
    'MaxEpochs',50, ...                % 最大迭代次数
     'GradientThreshold',1, ...         % 梯度阈值 
    'InitialLearnRate',pop(2));



% 训练LSTM
net = trainNetwork(inputn,outputn,layers,options);
% 预测
net = resetState(net);% 网络的更新状态可能对分类产生了负面影响。重置网络状态并再次预测序列。


[~,Ytrain]= predictAndUpdateState(net,XValidation);
cg = mse(Ytrain,YValidation);
 toc
disp('-------------------------')
result = cg;


end