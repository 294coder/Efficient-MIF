%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Adapting learning rate for new image dimension
% Interface:
%           lr = ft_learning_rate(model,net_scope,size_ms)
% Inputs:
%           model:        Struct with training details of pretrained network;
%           net_scope:    Network scope;
%           size_ms:      Size of ms image. 
% Output:
%           lr:           Learning rate for fine tuning.
% 
% References:
%           [Scarpa2018]    G. Scarpa, S. Vitale, and D. Cozzolino.
%                           Target-adaptive CNN-based pansharpening.
%                           IEEE Transactions on Geoscience and Remote Sensing, 
%                           vol. 56, no. 9, pp. 5443-5457, Sep. 2018.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lr = ft_learning_rate(model,net_scope,size_ms)
pretrained_lr = model.lr;
if isfield(model,'block_size')
        patch_size = double(model.block_size);
    else
        patch_size = double(model.patch_size);
end
lr = pretrained_lr*((patch_size-net_scope+1)^2./(size_ms(1)-net_scope+1)/(size_ms(1)-net_scope+1));
end
