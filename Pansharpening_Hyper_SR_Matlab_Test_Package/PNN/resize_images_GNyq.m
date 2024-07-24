%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%           Generate the low resolution PANchromatic (PAN) and MultiSpectral (MS) images according to Wald's protocol. 
%           
% Interface:
%           [I_MS_LR, I_PAN_LR] = resize_images_GNyq(I_MS,I_PAN,ratio,GNyqMS)
%
% Inputs:
%           I_MS:           MS image upsampled at PAN scale;
%           I_PAN:          PAN image;
%           ratio:          Scale ratio between MS and PAN. Pre-condition: Integer value;
%           GNyqMS:         (optional) custom MS MTF gains.
%
% Outputs:
%           I_MS_LR:        Low Resolution MS image;
%           I_PAN_LR:       Low Resolution PAN image.
% 
% References:
%           [Wald97]        L. Wald, T. Ranchin, and M. Mangolini, "Fusion of satellite images of different spatial resolutions: assessing the quality of resulting images,"
%                           Photogrammetric Engineering and Remote Sensing, vol. 63, no. 6, pp. 691-699, June 1997.
%           [Aiazzi06]      B. Aiazzi, L. Alparone, S. Baronti, A. Garzelli, and M. Selva, "MTF-tailored multiscale fusion of high-resolution MS and Pan imagery,"
%                           Photogrammetric Engineering and Remote Sensing, vol. 72, no. 5, pp. 591-596, May 2006.
%           [Vivone15]      G. Vivone, L. Alparone, J. Chanussot, M. Dalla Mura, A. Garzelli, G. Licciardi, R. Restaino, and L. Wald, “A Critical Comparison Among Pansharpening Algorithms”, 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 53, no. 5, pp. 2565–2586, May 2015.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [I_MS_LR, I_PAN_LR] = resize_images_GNyq(I_MS,I_PAN,ratio,GNyqMS)

I_MS = double(I_MS);
I_PAN = double(I_PAN);

%%% MTF
N = 41;
nBands = length(GNyqMS);
h = zeros(N, N, nBands);
fcut = 1/ratio;

for ii = 1 : nBands
    alpha = sqrt(((N-1)*(fcut/2))^2/(-2*log(GNyqMS(ii))));
    H = fspecial('gaussian', N, alpha);
    Hd = H./max(H(:));
    h(:,:,ii) = fwind1(Hd,kaiser(N));
end

I_MS_LP = zeros(size(I_MS));
for ii = 1 : size(I_MS,3)
    I_MS_LP(:,:,ii) = imfilter(I_MS(:,:,ii),real(h(:,:,ii)),'replicate');
end

%%% Decimation MS
I_MS_LP_D = zeros(round(size(I_MS,1)/ratio),round(size(I_MS,2)/ratio),size(I_MS,3));
for idim = 1 : size(I_MS,3)
    I_MS_LP_D(:,:,idim) = imresize(I_MS_LP(:,:,idim),1/ratio,'nearest');
end

I_MS_LR = double(I_MS_LP_D);

I_PAN_LR = imresize(I_PAN, 1/ratio);

end