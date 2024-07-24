%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Fine Tuning of Pretrained network for panshaprening
% Interface:
%           net = fine_tune(I_MS, I_PAN, model,layers,epochs,outpath,MTF)
%
% Inputs:
%           I_MS:     multispectral image;
%           I_PAN:    panchromatic image;
%           model:    struct with training details of pretrained network;
%           layers:   pretrained layers;
%           epochs:   number of epochs for fine tuning;
%           outpath:  path for storing the finetuned network;
%           MTF:      (optional) struct with custom MTF gains (GNyq and GNyqPan).
%
% Outputs:
%           net:        finetuned network;
%           traininfo:  training infos.
%           
% References:
%           [Scarpa2018]    G. Scarpa, S. Vitale, and D. Cozzolino.
%                           Target-adaptive CNN-based pansharpening.
%                           IEEE Transactions on Geoscience and Remote Sensing, 
%                           vol. 56, no. 9, pp. 5443-5457, Sep. 2018.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [net, traininfo] = fine_tune(I_MS, I_PAN, model,layers,epochs,outpath,MTF)

%% load data for fine tuning
net_scope = model_scope(model);
ratio = double(model.ratio);
sensor  = model.sensor;
s = size(I_MS);
lr = ft_learning_rate(model,net_scope,s); %adapting learning rate for new image dimension

%% Wald protocol and input preparation
if nargin == 7
    [I_MS_LR,I_PAN_LR] = resize_images_GNyq(I_MS,I_PAN,ratio,MTF.GNyq);
else
    [I_MS_LR,I_PAN_LR] = resize_images(I_MS,I_PAN,ratio,sensor);
end
I_MS = I_MS./(2^double(model.L));
I_in = input_preparation([],I_MS_LR,I_PAN_LR,model);
I_in = I_in(floor(net_scope/2)+1:end-floor(net_scope/2),floor(net_scope/2)+1:end-floor(net_scope/2),:,:);
I_ref = I_MS-I_in(:,:,1:end-1); %residual

%% training options
options = trainingOptions('sgdm',...
    'ExecutionEnvironment','auto',...
    'Momentum',0.9,...
    'VerboseFrequency',1,...
    'Shuffle','never',...
    'InitialLearnRate',lr,...
    'LearnRateDropPeriod',epochs+1,...
    'LearnRateDropFactor',1,...
    'L2Regularization',0.,...
    'MaxEpochs',epochs,...
    'CheckpointPath',outpath);

%% train network
[net, traininfo] = trainNetwork(I_in,I_ref(floor(net_scope/2)+1:end-floor(net_scope/2),floor(net_scope/2)+1:end-floor(net_scope/2),:),layers,options);
%save losses
%save([outpath,'traininfo.mat'],'traininfo');


end