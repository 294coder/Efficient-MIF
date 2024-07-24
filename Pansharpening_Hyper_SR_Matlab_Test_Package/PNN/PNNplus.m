%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Description: 
%        PNN+ algorithm for pansharpening by Scarpa et al. (2018).          
% Interface:
%         P      = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits, MTF, EXP);
%         P      = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits, MTF);
%         P      = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits);
%         P      = PNNplus(MS, PAN, sensor_model, FT_epochs);
%         P      = PNNplus(MS, PAN, sensor_model);
%        [P AUX] = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits, MTF, EXP);
%        [P AUX] = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits, MTF);
%        [P AUX] = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits);
%        [P AUX] = PNNplus(MS, PAN, sensor_model, FT_epochs);
%        [P AUX] = PNNplus(MS, PAN, sensor_model).
% Input:
%        MS:              4- or 8-band multispectral image;
%        PAN:             Panchromatic image (must be 4x4x larger than MS);
%        sensor_model:    one of the following string:
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
%                        provided model (e.g., 'IKONOS_PNNplus_model.mat').
%                        
%        nBits:           #bits (radiometric precision);  default=11;
%        FT_epochs(>=0):  0->no fine tuning;              default=50;
%               WARNING: In case of CPU-only PC it is recommended to avoid
%               (FT_epochs = 0) or limit to a few iterations the
%               fine-tuning.
%        MTF:             Struct with custom MTF gains (GNyq and GNyqPan). 
%        EXP:             4- or 8-band multispectral image upsampled to the PAN scale
% Output:
%        P:               Pansharpened image;
%        AUX:             Auxiliary output variable containing the
%                         fine-tuned model and other training details.
% References:
%           [Scarpa18]      G. Scarpa, S. Vitale, and D. Cozzolino, "Target-adaptive CNN-based pansharpening", 
%                           IEEE Transactions on Geoscience and Remote Sensing, vol. 56, no. 9, pp. 5443-5457, Sep. 2018.
%           [Vivone20]      G. Vivone, M. Dalla Mura, A. Garzelli, R. Restaino, G. Scarpa, M.O. Ulfarsson, L. Alparone, and J. Chanussot, "A New Benchmark Based on Recent Advances in Multispectral Pansharpening: Revisiting pansharpening with classical and emerging pansharpening methods", 
%                           IEEE Geoscience and Remote Sensing Magazine, doi: 10.1109/MGRS.2020.3019315.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [P, AUX] = PNNplus(MS, PAN, sensor_model, FT_epochs, nBits, MTF, EXP)

if nargin < 7
    EXP = [];
end

MTFflag = false; if ~isempty(MTF), MTFflag = true; end
if nargin<5, nBits=11; elseif isempty(nBits), nBits = 11; end 
if nargin<4, FT_epochs = 50; elseif isempty(FT_epochs), FT_epochs = 50; end

MS = single(MS); PAN = single(PAN); nBits = single(nBits); EXP = single(EXP);

[M,N,Nb] = size(MS);

%%% LOAD PRETRAINED MODEL
cd 'models'

available_models = {'IKONOS', 'GeoEye1','WV2', 'WV3'};
if sum(ismember(available_models,sensor_model)) 
    sensor_model = which([sensor_model '_PNNplus_model.mat']);
end

model = load(sensor_model);

cd ..

if isfield(model,'model'), model = model.model; end

model.L = nBits;

net_scope = model_scope(model);

% build list of layers
firstLayer = imageInputLayer([M N Nb+1],'Name','InputLayer','Normalization','none');
layers = load_layers(firstLayer,model);

%scaling learning rate on last layer
layers(end).WeightLearnRateFactor =0.1;     
layers(end).BiasLearnRateFactor =0.1;

layers = [layers,maeRegressionLayer('regre')];

traininfo = [];
pad = model.padSize;
[Mpan, Npan] = size(PAN);
% Train_time = 0;
if FT_epochs>0
    tempDir = 'temporary_pnn_plus';
    delete([tempDir '/*.*']);
    mkdir(tempDir);
    tic;
    if MTFflag
        [~, traininfo] = fine_tune(MS, PAN, model, layers, FT_epochs, tempDir, MTF);
    else
        [~, traininfo] = fine_tune(MS, PAN, model, layers, FT_epochs, tempDir);
    end
%     Train_time = toc;
    
    %%%% LOAD BEST MODEL
    [~, best] = min(traininfo.TrainingLoss);
    pref = [tempDir sprintf('/net_checkpoint__%d__*',best)];
    fn = dir(pref); fn = [tempDir '/' fn.name];
    net = getfield(load(fn),'net');
    %%%%
    
    delete([tempDir '/*.*']);  rmdir(tempDir);
    lgraph = layerGraph(net.Layers);
    model.layers{1} = net.Layers(2).Weights;
    model.layers{2} = net.Layers(2).Bias;
    model.layers{3} = net.Layers(4).Weights;
    model.layers{4} = net.Layers(4).Bias;
    model.layers{5} = net.Layers(6).Weights;
    model.layers{6} = net.Layers(6).Bias;        
else
    lgraph = layerGraph(layers);
end
lgraph = replaceLayer(lgraph,lgraph.Layers(1).Name,imageInputLayer([Mpan+pad Npan+pad Nb+1],'Name','InputLayer','Normalization','none'));
net = assembleNetwork(lgraph);
   
AUX.model = model;
AUX.traininfo = traininfo;

%% Pansharpening
I_in = input_preparation(EXP,MS,PAN,model);
tic;
P = predict(net,I_in);
%Test_time = toc;

I_MS_int = (2^model.L)*I_in(floor(net_scope/2)+1:end-floor(net_scope/2),floor(net_scope/2)+1:end-floor(net_scope/2),1:end-1,:);
P = P*(2^model.L)+I_MS_int;
P(P<0)=0;

%%%
%fprintf(sprintf('------>  [PNN+]: Fine-tuning (%d it) time = %0.4gs  //  Prediction time = %0.4gs\n',FT_epochs, Train_time, Test_time));

end