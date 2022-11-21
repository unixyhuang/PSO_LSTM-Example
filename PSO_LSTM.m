%% 粒子群优化LSTM(PSO_LSTM)
clc
clear
close  all
%% 数据读取
geshu=200;%训练集的个数
%读取数据
shuru=xlsread('input.xlsx');
shuchu=xlsread('output.xlsx');
nn = randperm(size(shuru,1));%随机排序
% nn=1:size(shuru,1);%正常排序
input_train =shuru(nn(1:geshu),:);
input_train=input_train';
output_train=shuchu(nn(1:geshu),:);
output_train=output_train';
input_test =shuru(nn((geshu+1):end),:);
input_test=input_test';
output_test=shuchu(nn((geshu+1):end),:);
output_test=output_test';
%样本输入输出数据归一化
[aa,bb]=mapminmax([input_train input_test]);
[cc,dd]=mapminmax([output_train output_test]);
global inputn outputn shuru_num shuchu_num XValidation YValidation
[inputn,inputps]=mapminmax('apply',input_train,bb);
[outputn,outputps]=mapminmax('apply',output_train,dd);
shuru_num = size(input_train,1); % 输入维度
shuchu_num = 1;  % 输出维度
dam = 10; % 验证集个数，验证集是从训练集里面取的数据
idx = randperm(size(inputn,2),dam);
XValidation = inputn(:,idx);
inputn(:,idx) = [];
YValidation = outputn(idx);
outputn(idx) = [];
YValidationy = output_train(idx);
output_train(idx) = [];
%%
% 1. 寻找最佳参数
NN=5;                   %初始化群体个数
D=2;                    %初始化群体维数，
T=10;                   %初始化群体最迭代次数
c1=2;                   %学习因子1
c2=2;                   %学习因子2
%用线性递减因子粒子群算法
Wmax=1.2; %惯性权重最大值
Wmin=0.8; %惯性权重最小值
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%每个变量的取值范围
ParticleScope(1,:)=[10 200];  % 中间层神经元个数
ParticleScope(2,:)=[0.01 0.15]; % 学习率
ParticleScope=ParticleScope';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
xv=rand(NN,2*D); %首先，初始化种群个体速度和位置
for d=1:D
    xv(:,d)=xv(:,d)*(ParticleScope(2,d)-ParticleScope(1,d))+ParticleScope(1,d);  
    xv(:,D+d)=(2*xv(:,D+d)-1 )*(ParticleScope(2,d)-ParticleScope(1,d))*0.2;
end
x1=xv(:,1:D);%位置
v1=xv(:,D+1:2*D);%速度
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------初始化个体位置和适应度值-----------------
p1=x1;
pbest1=ones(NN,1);
for i=1:NN
    pbest1(i)=fitness(x1(i,:));
end
%------初始时全局最优位置和最优值---------------
gbest1=min(pbest1);
lab=find(min(pbest1)==pbest1);
g1=x1(lab,:);
gb1=ones(1,T);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%-----浸入主循环，按照公式依次迭代直到迭代次数---
% N=40;                   %初始化群体个数
% D=10;                   %初始化群体维数
% T=100;                 %初始化群体最迭代次数
for i=1:T
    for j=1:NN
        if (fitness(x1(j,:))<pbest1(j))
            p1(j,:)=x1(j,:);%变量
            pbest1(j)=fitness(x1(j,:));
        end
        if(pbest1(j)<gbest1)
            g1=p1(j,:);%变量
            gbest1=pbest1(j);
        end
         w=Wmax-(Wmax-Wmin)*i/T;          
         v1(j,:)=w*v1(j,:)+c1*rand*(p1(j,:)-x1(j,:))+c2*rand*(g1-x1(j,:));
         x1(j,:)=x1(j,:)+v1(j,:); 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%位置约束
        label=find(x1(j,:)>ParticleScope(2,:));
        x1(j,label)=ParticleScope(2,label);        
        label2=find(x1(j,:)<ParticleScope(1,:));
        x1(j,label2)=ParticleScope(1,label2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%速度约束                
        labe3=find(v1(j,:)>ParticleScope(2,:)*0.2);
        v1(j,labe3)=ParticleScope(2,labe3)*0.2;        
        label4=find(v1(j,:)<-ParticleScope(2,:)*0.2);
        v1(j,label4)=-ParticleScope(2,label4)*0.2;              
    end
        %         gb1(i)=min(pbest1);
        gb1(i)=gbest1;
end
zhongjian1_num = round(g1(1));  
xue = g1(2);
%% 模型建立与训练
layers = [ ...
    sequenceInputLayer(shuru_num)    % 输入层
    lstmLayer(zhongjian1_num)        % LSTM层
    fullyConnectedLayer(shuchu_num)  % 全连接层
    regressionLayer];
 
options = trainingOptions('adam', ...  % 梯度下降
    'MaxEpochs',50, ...                % 最大迭代次数
    'GradientThreshold',1, ...         % 梯度阈值 
    'InitialLearnRate',xue,...
    'Verbose',0, ...
    'Plots','training-progress','ValidationData',{XValidation,YValidation});
%% 训练LSTM
net = trainNetwork(inputn,outputn,layers,options);
%% 预测
net = resetState(net);% 网络的更新状态可能对分类产生了负面影响。重置网络状态并再次预测序列。
[~,Ytrain]= predictAndUpdateState(net,inputn);
test_simu=mapminmax('reverse',Ytrain,dd);%反归一化
%测试集样本输入输出数据归一化
inputn_test=mapminmax('apply',input_test,bb);
[net,an]= predictAndUpdateState(net,inputn_test);
test_simu1=mapminmax('reverse',an,dd);%反归一化
error1=test_simu1-output_test;%测试集预测-真实
[~,Ytrain]= predictAndUpdateState(net,XValidation);
test_simuy=mapminmax('reverse',Ytrain,dd);%反归一化
%% 画图
figure
plot(output_train,'r-o','Color',[255 0 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[255 0 0]./255)
hold on
plot(test_simu,'-s','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',5,'MarkerFaceColor',[0 0 0]./255)
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("训练集")

figure
plot(YValidationy,'-o','Color',[255 255 0]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[255 0 0]./255)
hold on
plot(test_simuy,'-s','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',5,'MarkerFaceColor',[0 0 0]./255)
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("验证集")

figure
plot(output_test,'-o','Color',[0 0 255]./255,'linewidth',0.8,'Markersize',4,'MarkerFaceColor',[25 0 255]./255)
hold on
plot(test_simu1,'-s','Color',[0 0 0]./255,'linewidth',0.8,'Markersize',5,'MarkerFaceColor',[0 0 0]./255)
hold off
legend(["真实值" "预测值"])
xlabel("样本")
title("测试集")

 % 真实数据，行数代表特征数，列数代表样本数output_test = output_test;
T_sim_optimized = test_simu1;  % 仿真数据

num=size(output_test,2);%统计样本总数
error=T_sim_optimized-output_test;  %计算误差
mae=sum(abs(error))/num; %计算平均绝对误差
me=sum((error))/num; %计算平均绝对误差
mse=sum(error.*error)/num;  %计算均方误差
rmse=sqrt(mse);     %计算均方误差根
% R2=r*r;
tn_sim = T_sim_optimized';
tn_test =output_test';
N = size(tn_test,1);
R2=(N*sum(tn_sim.*tn_test)-sum(tn_sim)*sum(tn_test))^2/((N*sum((tn_sim).^2)-(sum(tn_sim))^2)*(N*sum((tn_test).^2)-(sum(tn_test))^2)); 

disp(' ')
disp('----------------------------------------------------------')

disp(['平均绝对误差mae为：            ',num2str(mae)])
disp(['平均误差me为：            ',num2str(me)])
disp(['均方误差根rmse为：             ',num2str(rmse)])
disp(['相关系数R2为：                ' ,num2str(R2)])







































