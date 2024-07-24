%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%        PNN algorithm for pansharpening by Masi et al. (2016). 
% Interface:
%         P      = PNN(MS, PAN, sensor_model, nBits, NDxI_flag, EXP);
%         P      = PNN(MS, PAN, sensor_model, nBits, NDxI_flag);
%         P      = PNN(MS, PAN, sensor_model, nBits);
%         P      = PNN(MS, PAN, sensor_model).
% Input:
%        MS:              4- or 8-band multispectral image;
%        PAN:             Panchromatic image (must be 4x4x larger than MS);
%        sensor_model:    One of the following string:
%                         'IKONOS'
%                         'GeoEye1'
%                         'WV2'
%                         'WV3'
%                         '<full-path file name of any pretrained model>'
%               WARNING: If an own model is used, the file must be a work
%                        space with a cell-array variable named 'layers'
%                        containing the sequence of weights and bias of the
%                        three convolutional layers. Also the following
%                        variables must be enclosed in the work space: lr, 
%                        patch_size, ratio, sensor, inputType, typeInterp.
%                        For consistency check watch the content of any 
%                        provided model (e.g., 'IKONOS_PNN_model.mat').
%                        
%        nBits:           #bits (radiometric precision).
%                         Default = 11;
%        NDxI_flag:       Boolean indicating the use of additional input
%                         bands such as NDVI, NDWI, etc. 
%                         Default = true;
%        EXP:             4- or 8-band multispectral image upsampled to the PAN scale
% Output:
%        P:               Pansharpened image;
% 
% References:
%           [Masi16]        G. Masi, D. Cozzolino, L. Verdoliva and G. Scarpa, "Pansharpening by Convolutional Neural Networks", 
%                           Remote Sensing 2016, 8(7):594.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function P = PNN(MS, PAN, sensor_model, nBits, NDxI_flag, EXP)

if nargin < 6
    EXP = [];
end

if nargin == 3, NDxI_flag = true; nBits = 11; end
if nargin == 5, if isempty(nBits), nBits = 11; end; end

MS = single(MS); PAN = single(PAN); nBits = single(nBits); EXP = single(EXP);

[M,N,Nb] = size(MS);

%%% LOAD PRETRAINED MODEL
available_models = {'IKONOS', 'GeoEye1','WV2', 'WV3'};
if sum(ismember(available_models,sensor_model))
    cd 'models'
    if NDxI_flag
        sensor_model = which([sensor_model '_PNN_model.mat']);
    else
        sensor_model = which([sensor_model '_PNN_noIDX_model.mat']);
    end
    cd ..
end

model = load(sensor_model);
model.L = nBits;

%net_scope = model_scope(model);

% build list of layers
if NDxI_flag
    if Nb == 8
        Nb = Nb+4; 
    else
        Nb = Nb+2; 
    end
end

firstLayer = imageInputLayer([M N Nb+1],'Name','InputLayer','Normalization','none');
layers = load_layers(firstLayer,model);

%scaling learning rate on last layer
layers(end).WeightLearnRateFactor =0.1; % Not needed     
layers(end).BiasLearnRateFactor =0.1; % Not needed

%layers = [layers,maeRegressionLayer('regre')];
layers = [layers, regressionLayer('Name','regre')]; % not needed

pad = model.padSize;
[Mpan, Npan] = size(PAN);
lgraph = layerGraph(layers);
lgraph = replaceLayer(lgraph,lgraph.Layers(1).Name,imageInputLayer([Mpan+pad Npan+pad Nb+1],'Name','InputLayer','Normalization','none'));
net = assembleNetwork(lgraph);

%% Pansharpening
I_in = input_preparation(EXP,MS,PAN,model);
tic;
P = predict(net,I_in);
%Test_time = toc;
P = P*(2^model.L);
P(P<0)=0;

%%%
%fprintf(sprintf('------>  [PNN]: Prediction time = %0.4gs\n', Test_time));

end