clear all;
 clc
 im = imread('./DRIVE/test/images/18_test.tif');
Truth = imread('./DRIVE/test/1st_manual/18_manual1.gif');
 TrainIDX =[];  TruthIDX = []; MaskIDX = [];
 Acc = zeros(20,1);
 for i = 1:20
    TrainIDX =  [TrainIDX; strcat(num2str(i+20),'_training','.tif')];
    TruthIDX = [TruthIDX;strcat(num2str(i+20),'_manual1','.gif')];
    MaskIDX = [MaskIDX; strcat(num2str(i+20),'_training_mask.gif')];
 end
 for counter = 1:20
    im = imread(TrainIDX(i,:));
    Truth = imread(TruthIDX(i,:));
% % % % % %     PreProcessing Function is called:
    Feature = Preprocessing(im,counter);
% % % % % % % % % % % % % 
    Truth = Truth(30:564, 15:550);
    Truth = double(Truth)/255;
    % % % % % % % % % % % % % % % % / PCA Implementaion
    [m, n, d] = size(Feature);
    Dimension = m*n;
    X = reshape(Feature, Dimension, d);
% % % % % %     The dataset will be saved X (matrix has 13 columns)
    str  = strcat('Input_features',num2str(counter),'.csv');
    Y = reshape(Truth,size(Truth,1) * size(Truth,2),1);
    [coeff,score] = princomp(X);
    % % % % % % % % % % % % % % % % % % % % % K-means Algorithms
     Res = kmeans(score,2,'distance','cosine');
    No_Cluster1 = length(Res==1);
    No_Cluster2 = length(Res==2);
    if (No_Cluster1>No_Cluster2)
        Pixel = 1;
    else
        Pixel = 2;
    end
    Res(Res==Pixel) = 0;
    Res(Res==(2 - Pixel + 1)) = 1;
    IDX = find(Res==0);
    XFinal = X(IDX,:);
    YFinal = Y(IDX);
    %%
     str = num2str(counter);
    path = strcat(str,'ELM.mat');
    
    Tr_LB = [XFinal,YFinal];
    [TrainingTime,TrainingAccuracy] = elm_train(Tr_LB, 1,311, 'sig',path)
    %%
    [TestingTime, TestingAccuracy ,Label] = elm_predict(Tr_LB,path);
    %%
    Res(IDX) = Label;    
    % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %  Post  Processing
    Temp = vec2mat(Res,size(Truth,1));
    FinalImage  = transpose(Temp);
    FinalImage = 1 - FinalImage;
    Label1 = length(find(FinalImage==1));
    Label0 = length(find(FinalImage==0));
    if (Label1>Label0)
        FinalImage = 1 - double(FinalImage);
    end
    [mask]= createmask(FinalImage);
     Final = (mask .*(FinalImage));
%     Final = mask - FinalImage;
    str  = strcat('Gabor_',num2str(counter),'.tif');
     imwrite(Final,str,'tif');
    % % % % % % % % % % /Evaluation Sections
    [m, n] = size(Final);
    Dimension = m*n;
    Predicted = reshape(Final, Dimension,1);
    Real = reshape(Truth, Dimension,1);
    TP = sum(Predicted==1 & Real==1);
    TN = sum(Predicted==0 & Real==0);
    FP = sum(Predicted==1 & Real==0);
    FN = sum(Predicted==0 & Real==1);
    Accuracy = (TP + TN)/(TP + TN + FP + FN)
    Acc(counter,1) = Accuracy;
    Sensitivity = TP/(TP + FN)
    Specificity = TN/(TN + FP)
    PPV = TP/(TP + FP)
 end
 [idx idy] = max(Acc)