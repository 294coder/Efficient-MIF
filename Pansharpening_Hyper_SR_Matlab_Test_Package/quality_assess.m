function [mean_psnr, mean_ssim] = quality_assess(imagery1, imagery2)
%==========================================================================
% Evaluates the quality assessment indices for two tensors.
%
% Syntax:
%   [mpsnr, mssim] = quality_access(imagery1, imagery2)
%
% Input:
%   imagery1 - the reference tensor
%   imagery2 - the target tensor

% NOTE: the tensor is a I1*I2*...*IN array and DYNAMIC RANGE [0, 255].
% Output:
%   mpsnr - Peak Signal-to-Noise Ratio
%   mssim - Structure SIMilarity
%==========================================================================
Nway = size(imagery1);
if length(Nway)>3
    imagery1 = reshape(imagery1,Nway(1),Nway(2),[]);
    imagery2 = reshape(imagery2,Nway(1),Nway(2),[]);
end
psnr = zeros(prod(Nway(3:end)),1);
ssim = psnr;
for ii = 1:prod(Nway(3:end))
    psnr(ii) = psnr_index(imagery1(:, :, ii), imagery2(:, :, ii));
    % 因为ssim_index是邓尚琦给的代码，需要乘255
    ssim(ii) = ssim_index(imagery1(:, :, ii)*255, imagery2(:, :, ii)*255);
end
mean_psnr = mean(psnr);
mean_ssim = mean(ssim);
%out = [mean(psnr), mean(ssim)];

