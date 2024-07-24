%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description:
%           Compute the network scope.
% Interface:
%           net_scope = model_scope(model)
% Input:
%           model:      Struct with training details of pretrained network.
% Output:
%           net_scope:  Scope of network.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function net_scope = model_scope(model)
    net_scope = 0;
    for i = 1:2:length(model.layers)
        net_scope = net_scope + size(model.layers{i},1)-1;
    end
    net_scope = double(net_scope+1);
end