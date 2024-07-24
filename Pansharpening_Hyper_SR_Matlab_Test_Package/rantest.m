%This is a demo to run fusion algorithms on Reduced Resolution
% LJ Deng(UESTC) TJ Zhang
% 2022-05-07
clear; close all;
%% =======load directors========
% Tools
addpath([pwd,'/Tools']);

% Select algorithms to run
algorithms = {'GT','EXP','BT-H','BDSD-PC','C-GSA','SR-D',...
    'MTF-GLP-HPM-R','MTF-GLP-FS','TV','PanNet','PNN','DiCNN','FusionNet','LAGConv','MSDCNN','BDPN'};%'PNN'
location1                = [2 40 4 43];  %default: data6: [10 50 1 60]; data7:[140 180 5 60]
location2                = [];  %default: data6: [190 240 5 60]; data7:[190 235 120 150]
    sensor = 'WV3';
%% =======read Multiple TestData_wv3.h5 (four 512x512 WV3 simulated data)========
file_test = '/Data2/DataSet/pansharpening_2/test_data/WV3/test_wv3_multiExm1.h5';
gt_multiExm_tmp = h5read(file_test,'/gt');  % WxHxCxN=1x2x3x4
gt_multiExm = permute(gt_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

ms_multiExm_tmp = h5read(file_test,'/ms');  % WxHxCxN=1x2x3x4
ms_multiExm = permute(ms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

lms_multiExm_tmp = h5read(file_test,'/lms');  % WxHxCxN=1x2x3x4
lms_multiExm = permute(lms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

pan_multiExm_tmp = h5read(file_test,'/pan');  % WxHxCxN=1x2x3x4
pan_multiExm = permute(pan_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

data_name = '3_EPS/WV3/multi/';  % director to save EPS figures


%% ==========Read each Data====================
exm_num = size(ms_multiExm, 1);
for i = 1 :   20% i = 1 or 2 ...    
    %% read each data
    HRMS_tmp = gt_multiExm(i, :, :, :); % I_GT
    I_GT     = squeeze(HRMS_tmp);
    LRMS_tmp = ms_multiExm(i, :, :, :); % I_MS_LR
    I_MS_LR     = squeeze(LRMS_tmp);
    LMS_tmp  = lms_multiExm(i, :, :, :); % I_MS
    I_MS     = squeeze(LMS_tmp);    
    PAN_tmp  = pan_multiExm(i, :, :, :); % I_PAN
    I_PAN      = squeeze(PAN_tmp); 

    %% Initialization of the Matrix of Results
    NumIndexes = 5;
    MatrixResults = zeros(numel(algorithms),NumIndexes);
    alg = 0;
    
    %% load Indexes for WV3_RR
    sensor = 'WV3';
    Qblocks_size = 32;
    bicubic = 0;% Interpolator
    flag_cut_bounds = 0;% Cut Final Image
    dim_cut = 21;% Cut Final Image
    thvalues = 0;% Threshold values out of dynamic range
    printEPS = 0;% Print Eps
    ratio = 4;% Resize Factor
    L = 11;% Radiometric Resolution
    
    %% show I_MS_LR, I_GT, PAN Imgs:
    showImage8_zoomin(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
   

    showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);
    print('-deps', strcat(data_name, num2str(i-1), '_pan', '.eps'))    
    
    %% ======GT ===================
    if ismember('GT',algorithms)
        alg = alg + 1;                                   
        [Q_avg_GT, SAM_GT, ERGAS_GT, SCC_GT, Q_GT] = indexes_evaluation(I_GT,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_GT,Q_avg_GT,SAM_GT,ERGAS_GT,SCC_GT];
        MatrixImage(:,:,:,alg) = I_GT;
        
        Q_avg_GT_multiexm(i) = Q_avg_GT;
        SAM_GT_multiexm(i)   = SAM_GT;
        ERGAS_GT_multiexm(i) = ERGAS_GT;
        SCC_GT_multiexm(i)   = SCC_GT;
        Q_GT_multiexm(i)     = Q_GT;
               
        showImage8_zoomin(I_GT,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
        print('-depsc', strcat(data_name, num2str(i-1), '_gt', '.eps'))
    end


    %% ====== 4) BDPN Method ======
    file_bdpn  = 'bdpn_wv3_rs';
    load(strcat('/Data2/DataSet/pansharpening_2/results/p2/wv3_multiExm1.h5/BDPN/Test/model_2022-05-21-10-46-03/results/output_mulExm_' , num2str(i-1), '.mat')) % load i-th image for DiCNN    
    I_bdpn  = double(sr);
    
    if ismember('BDPN',algorithms)
        alg = alg + 1;
        [Q_avg_bdpn, SAM_bdpn, ERGAS_bdpn, SCC_bdpn, Q_bdpn] = indexes_evaluation(I_bdpn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_bdpn,Q_avg_bdpn,SAM_bdpn,ERGAS_bdpn,SCC_bdpn];
        MatrixImage(:,:,:,alg) = I_bdpn;

        Q_avg_bdpn_multiexm(i) = Q_avg_bdpn;
        SAM_bdpn_multiexm(i)   = SAM_bdpn;
        ERGAS_bdpn_multiexm(i) = ERGAS_bdpn;
        SCC_bdpn_multiexm(i)   = SCC_bdpn;
        Q_bdpn_multiexm(i)     = Q_bdpn;  
        
        showImage8_zoomin(I_bdpn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
        print('-depsc', strcat(data_name, num2str(i-1),'_bdpn.eps'))
    end           
%     
%     
    
%     %% ====== 7) APNN Method ======
%     file_apnn = 'apnn_wv3_rs';
%     load(strcat('2_DL_Result/WV3/APNN/', file_apnn, num2str(i-1), '.mat')) % load i-th image for DiCNN
%     I_apnn = 2047*double(apnn_wv3_rs);
%     
%     if ismember('APNN',algorithms)
%         alg = alg + 1;
%         [Q_avg_apnn, SAM_apnn, ERGAS_apnn, SCC_apnn, Q_apnn] = indexes_evaluation(I_apnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%         MatrixResults(alg,:) = [Q_apnn,Q_avg_apnn,SAM_apnn,ERGAS_apnn,SCC_apnn];
%         MatrixImage(:,:,:,alg) = I_apnn;
%         
%         Q_avg_apnn_multiexm(i) = Q_avg_apnn;
%         SAM_apnn_multiexm(i)   = SAM_apnn;
%         ERGAS_apnn_multiexm(i) = ERGAS_apnn;
%         SCC_apnn_multiexm(i)   = SCC_apnn;
%         Q_apnn_multiexm(i)     = Q_apnn;  
%         
%         showImage8_zoomin(I_apnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%         print('-depsc', strcat(data_name, num2str(i-1),'_apnn.eps'))
%     end          
    
   
    
    
end

%% Print in LATEX

%% View All

if size(I_GT,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,2];
end

titleImages = algorithms;  
figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

%% ======Display the final average performance =======
% GT: average Q_avg

                      
% bdpn: average Q_avg
avg_Q_bdpn_multiexm = mean(Q_bdpn_multiexm);
std_Q_bdpn_multiexm = std(Q_bdpn_multiexm);

avg_Q_avg_bdpn_multiexm = mean(Q_avg_bdpn_multiexm);
std_Q_avg_bdpn_multiexm = std(Q_avg_bdpn_multiexm);

avg_SAM_bdpn_multiexm = mean(SAM_bdpn_multiexm);
std_SAM_bdpn_multiexm = std(SAM_bdpn_multiexm);

avg_ERGAS_bdpn_multiexm = mean(ERGAS_bdpn_multiexm);
std_ERGAS_bdpn_multiexm = std(ERGAS_bdpn_multiexm);

avg_SCC_bdpn_multiexm = mean(SCC_bdpn_multiexm);
std_SCC_bdpn_multiexm = std(SCC_bdpn_multiexm);

Avg_MatrixResults(17,:) = [avg_Q_bdpn_multiexm, std_Q_bdpn_multiexm, avg_Q_avg_bdpn_multiexm, std_Q_avg_bdpn_multiexm, ...
                          avg_SAM_bdpn_multiexm, std_SAM_bdpn_multiexm, avg_ERGAS_bdpn_multiexm, std_ERGAS_bdpn_multiexm,...
                          avg_SCC_bdpn_multiexm, std_SCC_bdpn_multiexm];                      


                                            
              

fprintf('\n')
disp('#######################################################')
disp(['Display the performance for:', num2str(1:i)])
disp('#######################################################')
disp(' |====Q====|===Q_avg===|=====SAM=====|======ERGAS=======|=======SCC=======')
MatrixResults