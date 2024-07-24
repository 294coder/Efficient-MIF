% LJ Deng(UESTC)
% 2020-06-02
clear; close all;
%% =======load directors========
% Tools
addpath([pwd,'/Tools']);
% L, locatio
% % Select algorithms to run
algorithms = {'BT-H'};%'PNN'

data_name = '3_EPS/WV3/wv3_os_';  % director to save EPS figures

%% ==========Read each Data====================
%% read each data
file_test = '/Data2/ZiHanCao/datasets/pansharpening/qb/full_examples/test_qb_OrigScale_multiExm1.h5';
disp(file_test)

%load(file_test)   % get I_MS_LR, I_MS, I_PAN and sensors' info.
ms_multiExm_tmp = h5read(file_test,'/ms');  % WxHxCxN=1x2x3x4
ms_multiExm = permute(ms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

lms_multiExm_tmp = h5read(file_test,'/lms');  % WxHxCxN=1x2x3x4
lms_multiExm = permute(lms_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3

pan_multiExm_tmp = h5read(file_test,'/pan');  % WxHxCxN=1x2x3x4
pan_multiExm = permute(pan_multiExm_tmp, [4 2 1 3]); % NxHxWxC=4x2x1x3
%% ==========Read each Data====================
exm_num = size(ms_multiExm, 1);
for i = (1: exm_num)
    disp(i)
    %% read each data
    LRMS_tmp = ms_multiExm(i, :, :, :); % I_MS_LR
    I_MS_LR     = squeeze(LRMS_tmp);
    LMS_tmp  = lms_multiExm(i, :, :, :); % I_MS
    I_MS     = squeeze(LMS_tmp);    
    PAN_tmp  = pan_multiExm(i, :, :, :); % I_PAN
    I_PAN      = squeeze(PAN_tmp); 

    NumIndexes = 3;
    MatrixResults = zeros(numel(algorithms),NumIndexes);
    alg = 0;

    flagQNR = 0; %% Flag QNR/HQNR, 1: QNR otherwise HQNR
 % for img show
    location1                = [10 50 190 240];  %default: data6: [10 50 1 60]; data7:[140 180 5 60]
    location2                = [230 280 240 290];  %default: data6: [190 240 5 60]; data7:[190 235 120 150]
    sensor = 'GF2';
    % disp(sensor)
    
%% load Indexes for WV3_FR
    % sensor = 'WV3';
     Qblocks_size = 32;
     bicubic = 0;% Interpolator
     flag_cut_bounds = 1;% Cut Final Image
     dim_cut = 21;% Cut Final Image
     thvalues = 0;% Threshold values out of dynamic range
     printEPS = 0;% Print Eps
    ratio = 4;% Resize Factor
    L = 11;% Radiometric Resolution
clear print

%% show I_MS_LR, I_GT, PAN Imgs:
% if size(I_MS,3) == 4
%     showImage4LR(I_MS_LR,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2); 
% else
%     showImage8_zoomin(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2); 
% end
% 
%     %showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);
%    showPan_zoomin(I_PAN,printEPS,2,flag_cut_bounds,dim_cut, location1, location2);
%    pause(2);print('-deps', strcat(data_name, num2str(i-1), '_pan', '.eps')) 


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% CS-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ====== 1) BT-H Method ======
    if ismember('BT-H',algorithms)
        alg = alg + 1;
        
        cd BT-H
        t2=tic;
        % I_BT_H = BroveyRegHazeMin(I_MS,I_PAN,ratio);
        % time_BT_H = toc(t2);
        % fprintf('Elaboration time BT-H: %.2f [sec]\n',time_BT_H);
        cd ..

        %%% Quality indexes computation                           
        [D_lambda_BT_H, D_S_BT_H, QNRI_BT_H] = indexes_evaluation_FS(I_MS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
        MatrixResults(alg,:) = [D_lambda_BT_H,D_S_BT_H,QNRI_BT_H];
        % MatrixImage(:,:,:,alg) = I_BT_H;
        
        D_lambda_BT_H_multiexm(i) = D_lambda_BT_H;
        D_S_BT_H_multiexm(i)   = D_S_BT_H;
        QNRI_BT_H_multiexm(i) = QNRI_BT_H;

%      
%        showImage8_zoomin(I_BT_H,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2); 
%         pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_bth.eps'))
    end

    %% ====== 2) BDSD-PC Method ======
    if ismember('BDSD-PC',algorithms)
        alg = alg + 1;

        cd BDSD
        t2=tic;
    I_BDSD_PC = BDSD_PC(I_MS,I_PAN,ratio,sensor);
    time_BDSD_PC = toc(t2);
    fprintf('Elaboration time BDSD-PC: %.2f [sec]\n',time_BDSD_PC);
    cd ..

        [D_lambda_BDSD_PC, D_S_BDSD_PC, QNRI_BDSD_PC] = indexes_evaluation_FS(I_BDSD_PC,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
        MatrixResults(alg,:) = [D_lambda_BDSD_PC,D_S_BDSD_PC,QNRI_BDSD_PC];
        MatrixImage(:,:,:,alg) = I_BDSD_PC;
        
        D_lambda_BDSD_PC_multiexm(i) = D_lambda_BDSD_PC;
        D_S_BDSD_PC_multiexm(i)   = D_S_BDSD_PC;
        QNRI_BDSD_PC_multiexm(i) = QNRI_BDSD_PC;
 
%          showImage8_zoomin(I_BDSD_PC,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%         pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_bdsd_pc.eps'))
    end


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%%%%%%%%% MRA-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%    %     
     %% ====== 1) MTF-GLP-HPM-R Method ======  
    if ismember('MTF-GLP-HPM-R',algorithms)
        alg = alg + 1;

        cd GLP
        t2=tic;
        I_MTF_GLP_HPM_R = MTF_GLP_HPM_R(I_MS,I_PAN,sensor,ratio);
        time_MTF_GLP_HPM_R = toc(t2);
        fprintf('Elaboration time MTF-GLP: %.2f [sec]\n',time_MTF_GLP_HPM_R);
        cd ..

        [D_lambda_MTF_GLP_HPM_R, D_S_MTF_GLP_HPM_R, QNRI_MTF_GLP_HPM_R] = indexes_evaluation_FS(I_MTF_GLP_HPM_R,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
        MatrixResults(alg,:) = [D_lambda_MTF_GLP_HPM_R,D_S_MTF_GLP_HPM_R,QNRI_MTF_GLP_HPM_R];
        MatrixImage(:,:,:,alg) = I_MTF_GLP_HPM_R;
        
        D_lambda_MTF_GLP_HPM_R_multiexm(i) = D_lambda_MTF_GLP_HPM_R;
        D_S_MTF_GLP_HPM_R_multiexm(i)   = D_S_MTF_GLP_HPM_R;
        QNRI_MTF_GLP_HPM_R_multiexm(i) = QNRI_MTF_GLP_HPM_R;
        
%        showImage8_zoomin(I_MTF_GLP_HPM_R,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2); 
%          pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_mtfglp_hpm_r.eps'))        

    end    
    


        
    %% ====== 2) MTF-GLP-FS Method ======  
    if ismember('MTF-GLP-FS',algorithms)
        alg = alg + 1;

        cd GLP
        t2=tic;
        I_MTF_GLP_FS = MTF_GLP_FS(I_MS,I_PAN,sensor,ratio);
        time_MTF_GLP_FS = toc(t2);
        fprintf('Elaboration time MTF-GLP-FS: %.2f [sec]\n',time_MTF_GLP_FS);
        cd ..

        [D_lambda_MTF_GLP_FS, D_S_MTF_GLP_FS, QNRI_MTF_GLP_FS] = indexes_evaluation_FS(I_MTF_GLP_FS,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
        MatrixResults(alg,:) = [D_lambda_MTF_GLP_FS,D_S_MTF_GLP_FS,QNRI_MTF_GLP_FS];
        MatrixImage(:,:,:,alg) = I_MTF_GLP_FS;
        
        D_lambda_MTF_GLP_FS_multiexm(i) = D_lambda_MTF_GLP_FS;
        D_S_MTF_GLP_FS_multiexm(i)   = D_S_MTF_GLP_FS;
         QNRI_MTF_GLP_FS_multiexm(i) = QNRI_MTF_GLP_FS;
%          showImage8_zoomin(I_MTF_GLP_FS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2); 
%         pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_mtfglpfs.eps'))               
    end   
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% VO-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ====== 1) TV Method ======
    if ismember('TV',algorithms)
        alg = alg + 1;
        %%%%%%%%%%%%%%%%%%%%%%%%%% Parameters setting %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        switch sensor
            case 'IKONOS'
                w=[0.1091    0.2127    0.2928    0.3854];
                c = 8;
                alpha=1.064;
                maxiter=10;
                lambda = 0.47106;
            case {'GeoEye1','WV4'}
                w=[0.1552, 0.3959, 0.2902, 0.1587];
                c = 8;
                alpha=0.75;
                maxiter=50;
                lambda = 157.8954;
            case 'WV3'
                w=[0.0657    0.1012    0.1537    0.1473    0.1245    0.1545    0.1338    0.1192];
                c = 8;
                alpha=0.75;
                maxiter=50;
                lambda = 1.0000e-03;
        end
        cd TV
        t2 = tic;
        I_TV = TV_pansharpen(I_MS_LR,I_PAN,alpha,lambda,c,maxiter,w);
        time_TV = toc(t2);
        fprintf('Elaboration time TV: %.2f [sec]\n',time_TV);
        cd ..

        [D_lambda_TV, D_S_TV, QNRI_TV] = indexes_evaluation_FS(I_TV,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
        MatrixResults(alg,:) = [D_lambda_TV,D_S_TV,QNRI_TV];
        MatrixImage(:,:,:,alg) = I_TV;

        D_lambda_TV_multiexm(i) = D_lambda_TV;
        D_S_TV_multiexm(i)   = D_S_TV;
        QNRI_TV_multiexm(i) = QNRI_TV;  
%         
%          showImage8_zoomin(I_TV,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2); 
%         pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_tv.eps'))               
    end    

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% DL-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
%   %% ====== 1) PNN Method ======
%    if ismember('PNN',algorithms)
%
%    load(strcat('2_DL_Result/WV3_Full/PNN/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for PNN
%    I_pnn = double(sr);
%
%
%        alg = alg + 1;
%        [D_lambda_pnn, D_S_pnn, QNRI_pnn] = indexes_evaluation_FS(I_pnn,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_pnn,D_S_pnn,QNRI_pnn];
%        MatrixImage(:,:,:,alg) = I_pnn;
%
%        D_lambda_pnn_multiexm(i) = D_lambda_pnn;
%        D_S_pnn_multiexm(i)   = D_S_pnn;
%        QNRI_pnn_multiexm(i) = QNRI_pnn;
%%
%%         showImage8_zoomin(I_pnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%%          pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_pnn.eps'))
%    end
%
%    %% ====== 2) PanNet Method ======
%
%    if ismember('PanNet',algorithms)
%    load(strcat('2_DL_Result/WV3_Full/PanNet/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for PanNet
%    I_pannet = double(sr);
%
%        alg = alg + 1;
%        [D_lambda_pannet, D_S_pannet, QNRI_pannet] = indexes_evaluation_FS(I_pannet,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_pannet,D_S_pannet,QNRI_pannet];
%        MatrixImage(:,:,:,alg) = I_pannet;
%
%
%        D_lambda_pannet_multiexm(i) = D_lambda_pannet;
%        D_S_pannet_multiexm(i)   = D_S_pannet;
%        QNRI_pannet_multiexm(i) = QNRI_pannet;
%%
%         showImage8_zoomin(I_pannet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_pannet.eps'))
%    end
%
%
%
%%% ====== 3) DiCNN Method ======
% if ismember('DiCNN',algorithms)
%    load(strcat('2_DL_Result/WV3_Full/DiCNN/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for DiCNN
%    I_dicnn = double(sr);
%
%
%        alg = alg + 1;
%        [D_lambda_dicnn, D_S_dicnn, QNRI_dicnn] = indexes_evaluation_FS(I_dicnn,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_dicnn,D_S_dicnn,QNRI_dicnn];
%        MatrixImage(:,:,:,alg) = I_dicnn;
%
%        D_lambda_dicnn_multiexm(i) = D_lambda_dicnn;
%        D_S_dicnn_multiexm(i)   = D_S_dicnn;
%        QNRI_dicnn_multiexm(i) = QNRI_dicnn;
%%
%         showImage8_zoomin(I_dicnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_dicnn.eps'))
% end
%    %% ====== 4) MSDCNN Method ======
%
%if ismember('MSDCNN',algorithms)
%    load(strcat('2_DL_Result/WV3_Full/MSDCNN/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for MSDCNN
%    I_msdcnn = double(sr);
%
%
%        alg = alg + 1;
%        [D_lambda_msdcnn, D_S_msdcnn, QNRI_msdcnn] = indexes_evaluation_FS(I_msdcnn,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_msdcnn,D_S_msdcnn,QNRI_msdcnn];
%        MatrixImage(:,:,:,alg) = I_msdcnn;
%
%        D_lambda_msdcnn_multiexm(i) = D_lambda_msdcnn;
%        D_S_msdcnn_multiexm(i)   = D_S_msdcnn;
%          QNRI_msdcnn_multiexm(i) = QNRI_msdcnn;
%          showImage8_zoomin(I_msdcnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%          pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_msdcnn.eps'))
%end
%    %% ====== 5) BDPN Method ======
%
%
%    if ismember('BDPN',algorithms)
%    load(strcat('2_DL_Result/WV3_Full/BDPN/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for BDPN
%    I_bdpn  = double(sr);
%
%
%        alg = alg + 1;
%        [D_lambda_bdpn, D_S_bdpn, QNRI_bdpn] = indexes_evaluation_FS(I_bdpn,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_bdpn,D_S_bdpn,QNRI_bdpn];
%        MatrixImage(:,:,:,alg) = I_bdpn;
%
%        D_lambda_bdpn_multiexm(i) = D_lambda_bdpn;
%        D_S_bdpn_multiexm(i)   = D_S_bdpn;
%        QNRI_bdpn_multiexm(i) = QNRI_bdpn;
%
%
%        showImage8_zoomin(I_bdpn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_bdpn.eps'))
%     end
%        %% ====== 6) FusionNet Method ======
%    if ismember('FusionNet',algorithms)
%    file_fusionnet = 'fusionnet_wv3_rs';
%    load(strcat('2_DL_Result/WV3_Full/FusionNet/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for FusionNet
%    I_fusionnet = double(sr);
%
%
%        alg = alg + 1;
%        [D_lambda_fusionnet, D_S_fusionnet, QNRI_fusionnet] = indexes_evaluation_FS(I_fusionnet,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_fusionnet,D_S_fusionnet,QNRI_fusionnet];
%        MatrixImage(:,:,:,alg) = I_fusionnet;
%
%        D_lambda_fusionnet_multiexm(i) = D_lambda_fusionnet;
%        D_S_fusionnet_multiexm(i)   = D_S_fusionnet;
%        QNRI_fusionnet_multiexm(i) = QNRI_fusionnet;
%
%        showImage8_zoomin(I_fusionnet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%       pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_fusionnet.eps'))
%    end
%        %% ====== 7) LAGConv Method ======
%
%    if ismember('LAGConv',algorithms)
%    load(strcat('2_DL_Result/WV3_Full/LAGConv/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for LAGConv
%    I_lagnet = double(sr);
%        alg = alg + 1;
%        [D_lambda_lagnet, D_S_lagnet, QNRI_lagnet] = indexes_evaluation_FS(I_lagnet,I_MS_LR,I_PAN,L,thvalues,I_MS,sensor,ratio,flagQNR);
%        MatrixResults(alg,:) = [D_lambda_lagnet,D_S_lagnet,QNRI_lagnet];
%        MatrixImage(:,:,:,alg) = I_lagnet;
%
%        D_lambda_lagnet_multiexm(i) = D_lambda_lagnet;
%        D_S_lagnet_multiexm(i)   = D_S_lagnet;
%        QNRI_lagnet_multiexm(i) = QNRI_lagnet;
%%
%        showImage8_zoomin(I_lagnet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_lagnet.eps'))
%    end

    
end
%% Print in LATEX
% if flagQNR == 1
%     matrix2latex(MatrixResults,'FR_Assessment.tex', 'rowLabels',algorithms,'columnLabels',[{'DL'},{'DS'},{'QNR'}],'alignment','c','format', '%.4f');
% else
%     matrix2latex(MatrixResults,'FR_Assessment.tex', 'rowLabels',algorithms,'columnLabels',[{'DL'},{'DS'},{'HQNR'}],'alignment','c','format', '%.4f');
% end

%% View All

if size(I_MS,3) == 4
    vect_index_RGB = [3,2,1];
else
    vect_index_RGB = [5,3,2];
end

titleImages = algorithms;
% figure, showImagesAll(MatrixImage,titleImages,vect_index_RGB,flag_cut_bounds,dim_cut,0);

                     
% BT_H: average D_lambda_avg
avg_D_lambda_BT_H_multiexm = mean(D_lambda_BT_H_multiexm);
std_D_lambda_BT_H_multiexm = std(D_lambda_BT_H_multiexm);

avg_D_S_BT_H_multiexm = mean(D_S_BT_H_multiexm);
std_D_S_BT_H_multiexm = std(D_S_BT_H_multiexm);

avg_QNRI_BT_H_multiexm = mean(QNRI_BT_H_multiexm);
std_QNRI_BT_H_multiexm = std(QNRI_BT_H_multiexm);

Avg_MatrixResults(1,:) = [avg_D_lambda_BT_H_multiexm, std_D_lambda_BT_H_multiexm, ...
                          avg_D_S_BT_H_multiexm, std_D_S_BT_H_multiexm, avg_QNRI_BT_H_multiexm, std_QNRI_BT_H_multiexm
                          ];                      
                      
% BDSD: average D_lambda_avg
% avg_D_lambda_BDSD_PC_multiexm = mean(D_lambda_BDSD_PC_multiexm);
% std_D_lambda_BDSD_PC_multiexm = std(D_lambda_BDSD_PC_multiexm);

% avg_D_S_BDSD_PC_multiexm = mean(D_S_BDSD_PC_multiexm);
% std_D_S_BDSD_PC_multiexm = std(D_S_BDSD_PC_multiexm);

% avg_QNRI_BDSD_PC_multiexm = mean(QNRI_BDSD_PC_multiexm);
% std_QNRI_BDSD_PC_multiexm = std(QNRI_BDSD_PC_multiexm);


% Avg_MatrixResults(2,:) = [avg_D_lambda_BDSD_PC_multiexm, std_D_lambda_BDSD_PC_multiexm, ...
%                           avg_D_S_BDSD_PC_multiexm, std_D_S_BDSD_PC_multiexm, avg_QNRI_BDSD_PC_multiexm, std_QNRI_BDSD_PC_multiexm
%                          ];   
                      
            
% % MTF_GLP: average D_lambda_avg
% avg_D_lambda_MTF_GLP_HPM_R_multiexm = mean(D_lambda_MTF_GLP_HPM_R_multiexm);
% std_D_lambda_MTF_GLP_HPM_R_multiexm = std(D_lambda_MTF_GLP_HPM_R_multiexm);

% avg_D_S_MTF_GLP_HPM_R_multiexm = mean(D_S_MTF_GLP_HPM_R_multiexm);
% std_D_S_MTF_GLP_HPM_R_multiexm = std(D_S_MTF_GLP_HPM_R_multiexm);

% avg_QNRI_MTF_GLP_HPM_R_multiexm = mean(QNRI_MTF_GLP_HPM_R_multiexm);
% std_QNRI_MTF_GLP_HPM_R_multiexm = std(QNRI_MTF_GLP_HPM_R_multiexm);

% Avg_MatrixResults(3,:) = [avg_D_lambda_MTF_GLP_HPM_R_multiexm, std_D_lambda_MTF_GLP_HPM_R_multiexm, ...
%                           avg_D_S_MTF_GLP_HPM_R_multiexm, std_D_S_MTF_GLP_HPM_R_multiexm, avg_QNRI_MTF_GLP_HPM_R_multiexm, std_QNRI_MTF_GLP_HPM_R_multiexm
%                      ];                        
                      
                      
                      
% % MTF_GLP_FS: average D_lambda_avg
% avg_D_lambda_MTF_GLP_FS_multiexm = mean(D_lambda_MTF_GLP_FS_multiexm);
% std_D_lambda_MTF_GLP_FS_multiexm = std(D_lambda_MTF_GLP_FS_multiexm);

% avg_D_S_MTF_GLP_FS_multiexm = mean(D_S_MTF_GLP_FS_multiexm);
% std_D_S_MTF_GLP_FS_multiexm = std(D_S_MTF_GLP_FS_multiexm);

% avg_QNRI_MTF_GLP_FS_multiexm = mean(QNRI_MTF_GLP_FS_multiexm);
% std_QNRI_MTF_GLP_FS_multiexm = std(QNRI_MTF_GLP_FS_multiexm);

% Avg_MatrixResults(4,:) = [avg_D_lambda_MTF_GLP_FS_multiexm, std_D_lambda_MTF_GLP_FS_multiexm, ...
%                           avg_D_S_MTF_GLP_FS_multiexm, std_D_S_MTF_GLP_FS_multiexm, avg_QNRI_MTF_GLP_FS_multiexm, std_QNRI_MTF_GLP_FS_multiexm
%                         ];                         
                      
                      
% % TV: average D_lambda_avg
% avg_D_lambda_TV_multiexm = mean(D_lambda_TV_multiexm);
% std_D_lambda_TV_multiexm = std(D_lambda_TV_multiexm);

% avg_D_S_TV_multiexm = mean(D_S_TV_multiexm);
% std_D_S_TV_multiexm = std(D_S_TV_multiexm);

% avg_QNRI_TV_multiexm = mean(QNRI_TV_multiexm);
% std_QNRI_TV_multiexm = std(QNRI_TV_multiexm);


% Avg_MatrixResults(5,:) = [avg_D_lambda_TV_multiexm, std_D_lambda_TV_multiexm, ...
%                           avg_D_S_TV_multiexm, std_D_S_TV_multiexm, avg_QNRI_TV_multiexm, std_QNRI_TV_multiexm];   
%% pnn: average D_lambda_avg
%avg_D_lambda_pnn_multiexm = mean(D_lambda_pnn_multiexm);
%std_D_lambda_pnn_multiexm = std(D_lambda_pnn_multiexm);
%
%avg_D_S_pnn_multiexm = mean(D_S_pnn_multiexm);
%std_D_S_pnn_multiexm = std(D_S_pnn_multiexm);
%
%avg_QNRI_pnn_multiexm = mean(QNRI_pnn_multiexm);
%std_QNRI_pnn_multiexm = std(QNRI_pnn_multiexm);
%
%
%Avg_MatrixResults(6,:) = [avg_D_lambda_pnn_multiexm, std_D_lambda_pnn_multiexm, ...
%                          avg_D_S_pnn_multiexm, std_D_S_pnn_multiexm, avg_QNRI_pnn_multiexm, std_QNRI_pnn_multiexm
%                         ];
%% pannet: average D_lambda_avg
%avg_D_lambda_pannet_multiexm = mean(D_lambda_pannet_multiexm);
%std_D_lambda_pannet_multiexm = std(D_lambda_pannet_multiexm);
%
%
%avg_D_S_pannet_multiexm = mean(D_S_pannet_multiexm);
%std_D_S_pannet_multiexm = std(D_S_pannet_multiexm);
%
%avg_QNRI_pannet_multiexm = mean(QNRI_pannet_multiexm);
%std_QNRI_pannet_multiexm = std(QNRI_pannet_multiexm);
%
%
%Avg_MatrixResults(7,:) = [avg_D_lambda_pannet_multiexm, std_D_lambda_pannet_multiexm, ...
%                          avg_D_S_pannet_multiexm, std_D_S_pannet_multiexm, avg_QNRI_pannet_multiexm, std_QNRI_pannet_multiexm
%                       ];
%
%
%
%% dicnn: average D_lambda_avg
%avg_D_lambda_dicnn_multiexm = mean(D_lambda_dicnn_multiexm);
%std_D_lambda_dicnn_multiexm = std(D_lambda_dicnn_multiexm);
%
%avg_D_S_dicnn_multiexm = mean(D_S_dicnn_multiexm);
%std_D_S_dicnn_multiexm = std(D_S_dicnn_multiexm);
%
%avg_QNRI_dicnn_multiexm = mean(QNRI_dicnn_multiexm);
%std_QNRI_dicnn_multiexm = std(QNRI_dicnn_multiexm);
%
%Avg_MatrixResults(8,:) = [avg_D_lambda_dicnn_multiexm, std_D_lambda_dicnn_multiexm, ...
%                          avg_D_S_dicnn_multiexm, std_D_S_dicnn_multiexm, avg_QNRI_dicnn_multiexm, std_QNRI_dicnn_multiexm
%                          ];
%% msdcnn: average D_lambda_avg
%avg_D_lambda_msdcnn_multiexm = mean(D_lambda_msdcnn_multiexm);
%std_D_lambda_msdcnn_multiexm = std(D_lambda_msdcnn_multiexm);
%
%
%avg_D_S_msdcnn_multiexm = mean(D_S_msdcnn_multiexm);
%std_D_S_msdcnn_multiexm = std(D_S_msdcnn_multiexm);
%
%avg_QNRI_msdcnn_multiexm = mean(QNRI_msdcnn_multiexm);
%std_QNRI_msdcnn_multiexm = std(QNRI_msdcnn_multiexm);
%
%Avg_MatrixResults(9,:) = [avg_D_lambda_msdcnn_multiexm, std_D_lambda_msdcnn_multiexm,  ...
%                          avg_D_S_msdcnn_multiexm, std_D_S_msdcnn_multiexm, avg_QNRI_msdcnn_multiexm, std_QNRI_msdcnn_multiexm
%                       ];
%% bdpn: average D_lambda_avg
%avg_D_lambda_bdpn_multiexm = mean(D_lambda_bdpn_multiexm);
%std_D_lambda_bdpn_multiexm = std(D_lambda_bdpn_multiexm);
%
%
%avg_D_S_bdpn_multiexm = mean(D_S_bdpn_multiexm);
%std_D_S_bdpn_multiexm = std(D_S_bdpn_multiexm);
%
%avg_QNRI_bdpn_multiexm = mean(QNRI_bdpn_multiexm);
%std_QNRI_bdpn_multiexm = std(QNRI_bdpn_multiexm);
%
%Avg_MatrixResults(10,:) = [avg_D_lambda_bdpn_multiexm, std_D_lambda_bdpn_multiexm,  ...
%                          avg_D_S_bdpn_multiexm, std_D_S_bdpn_multiexm, avg_QNRI_bdpn_multiexm, std_QNRI_bdpn_multiexm
%                       ];
%
%
%
%
%%
%% fusionnet: average D_lambda_avg
%avg_D_lambda_fusionnet_multiexm = mean(D_lambda_fusionnet_multiexm);
%std_D_lambda_fusionnet_multiexm = std(D_lambda_fusionnet_multiexm);
%
%avg_D_S_fusionnet_multiexm = mean(D_S_fusionnet_multiexm);
%std_D_S_fusionnet_multiexm = std(D_S_fusionnet_multiexm);
%
%avg_QNRI_fusionnet_multiexm = mean(QNRI_fusionnet_multiexm);
%std_QNRI_fusionnet_multiexm = std(QNRI_fusionnet_multiexm);
%
%Avg_MatrixResults(11,:) = [avg_D_lambda_fusionnet_multiexm, std_D_lambda_fusionnet_multiexm, ...
%                          avg_D_S_fusionnet_multiexm, std_D_S_fusionnet_multiexm, avg_QNRI_fusionnet_multiexm, std_QNRI_fusionnet_multiexm
%                         ];
%% lagnet: average D_lambda_avg
%avg_D_lambda_lagnet_multiexm = mean(D_lambda_lagnet_multiexm);
%std_D_lambda_lagnet_multiexm = std(D_lambda_lagnet_multiexm);
%
%avg_D_S_lagnet_multiexm = mean(D_S_lagnet_multiexm);
%std_D_S_lagnet_multiexm = std(D_S_lagnet_multiexm);
%
%avg_QNRI_lagnet_multiexm = mean(QNRI_lagnet_multiexm);
%std_QNRI_lagnet_multiexm = std(QNRI_lagnet_multiexm);
%
%
%Avg_MatrixResults(12,:) = [avg_D_lambda_lagnet_multiexm, std_D_lambda_lagnet_multiexm, ...
%                          avg_D_S_lagnet_multiexm, std_D_S_lagnet_multiexm, avg_QNRI_lagnet_multiexm, std_QNRI_lagnet_multiexm
%                          ];
                                            
              
                      
matrix2latex(Avg_MatrixResults,'Avg_RR_Assessment.tex', 'rowLabels',algorithms,'columnLabels',[{'D_lambda'}, {'D_l-std'}, {'D_S'}, {'D_S-std'}, {'QNRI'}, {'QNRI-std'}],'alignment','c','format', '%.4f');

fprintf('\n')
disp('#######################################################')
disp(['Display the performance for:', num2str(1:i)])
disp('#######################################################')
disp(' |===D_lambda_avg===|=====D_s=====|======QNR=======')
Avg_MatrixResults
