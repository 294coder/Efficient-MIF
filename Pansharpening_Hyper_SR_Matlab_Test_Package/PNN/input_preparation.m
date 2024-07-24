%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description:
%           Input preparation 
% Interface:
%           I_in = input_preparation(I_MS_LR,I_PAN,model)
% Inputs:
%           EXP:        4- or 8-band multispectral image upsampled to the PAN scale;
%           I_MS_LR:    MS image donwgraded for Wald Protocol;
%           I_PAN:      PAN image downgraded for Wald Protocol;
%           model:      Struct with training details of pretrained network.
% Output:
%           I_in:       Stack of (MS,PAN) for fine tune the network.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function I_in = input_preparation(EXP,I_MS_LR,I_PAN,model)

I_MS_LR = double(I_MS_LR);
I_PAN = double(I_PAN);
NDxI_LR = [];

if isequal(model.inputType,'MS_PAN_NDxI')
    if size(I_MS_LR,3) == 8
        NDxI_LR = cat(3,...
                  (I_MS_LR(:,:,5)-I_MS_LR(:,:,8))./(I_MS_LR(:,:,5)+I_MS_LR(:,:,8)), ...
                  (I_MS_LR(:,:,1)-I_MS_LR(:,:,8))./(I_MS_LR(:,:,1)+I_MS_LR(:,:,8)), ...
                  (I_MS_LR(:,:,3)-I_MS_LR(:,:,4))./(I_MS_LR(:,:,3)+I_MS_LR(:,:,4)), ...
                  (I_MS_LR(:,:,6)-I_MS_LR(:,:,1))./(I_MS_LR(:,:,6)+I_MS_LR(:,:,1)) );
    else
        NDxI_LR = cat(3,...
                  (I_MS_LR(:,:,4)-I_MS_LR(:,:,3))./(I_MS_LR(:,:,4)+I_MS_LR(:,:,3)), ...
                  (I_MS_LR(:,:,2)-I_MS_LR(:,:,4))./(I_MS_LR(:,:,2)+I_MS_LR(:,:,4)) );
    end
    NDxI_LR(find(isnan(NDxI_LR))) = 0;
end

mav_value = 2^double(model.L);

if ~isempty(EXP)
    I_MS = EXP;
    if not(isempty(NDxI_LR)), NDxI = interp23tap(NDxI_LR,model.ratio); end
else
    if isequal(model.typeInterp,'interp23tap')
        I_MS = interp23tap(I_MS_LR, double(model.ratio));
        if not(isempty(NDxI_LR)), NDxI = interp23tap(NDxI_LR,model.ratio); end
    elseif isequal(model.typeInterp,'cubic')
        I_MS = imresize(I_MS_LR,size(I_PAN),'bicubic');
        if not(isempty(NDxI_LR)), NDxI = imresize(NDxI_LR,size(I_PAN),'bicubic'); end
    else
        error('Interpolation not supported');
    end
end

if isequal(model.inputType,'MS')
    I_in = single(I_MS)/mav_value;
elseif isequal(model.inputType,'MS_PAN')
    I_in = single(cat(3,I_MS,I_PAN))/mav_value;
elseif isequal(model.inputType,'MS_PAN_NDxI')
    I_in = single(cat(3,I_MS,I_PAN))/mav_value;
    I_in = single(cat(3,I_in,single(NDxI)));
else
   error('Configuration not supported');
end

padSize = double(model.padSize);
I_in = padarray(I_in, [padSize,padSize]/2, 'replicate','both');

end
