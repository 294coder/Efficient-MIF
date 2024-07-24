%This is a demo to run fusion algorithms on Reduced Resolution
% LJ Deng(UESTC)
% 2022-06-02
clear; close all;
%% =======load directors========
% Tools
addpath([pwd,'/Tools']);

% Select algorithms to run
algorithms = {'GT'};%'PNN'
location1                = [40  60 4 43];  % Location of zoom in
location2                = []; 
%sensor = 'WV3';
%% =======read Multiple TestData_wv3.h5 (four 512x512 WV3 simulated data)========
file_test = '/Data2/ZiHanCao/datasets/pansharpening/qb/reduced_examples/test_qb_multiExm1.h5';
disp(file_test)
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
for i = 1 : exm_num  % i = 1 or 2 ...    
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
    sensor = 'GF2';
    % disp(sensor)
    Qblocks_size = 32;
    bicubic = 0;% Interpolator
    flag_cut_bounds = 1;% Cut Final Image
    dim_cut = 30;% Cut Final Image
    thvalues = 0;% Threshold values out of dynamic range
    printEPS = 0;% Print Eps
    ratio = 4;% Resize Factor
    L = 11;% Radiometric Resolution
    
    %% show I_MS_LR, I_GT, PAN Imgs:
%    showImage8_zoomin(I_MS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
   
 
%    showPan(I_PAN,printEPS,2,flag_cut_bounds,dim_cut);
%    pause(2);print('-deps', strcat(data_name, num2str(i-1), '_pan', '.eps'))
    
    %% ======GT ===================
    if ismember('GT',algorithms)
        alg = alg + 1;                                   
        [Q_avg_GT, SAM_GT, ERGAS_GT, SCC_GT, Q_GT] = indexes_evaluation(I_GT,I_MS,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_GT,Q_avg_GT,SAM_GT,ERGAS_GT,SCC_GT];
        MatrixImage(:,:,:,alg) = I_GT;
        
        Q_avg_GT_multiexm(i) = Q_avg_GT;
        SAM_GT_multiexm(i)   = SAM_GT;
        ERGAS_GT_multiexm(i) = ERGAS_GT;
        SCC_GT_multiexm(i)   = SCC_GT;
        Q_GT_multiexm(i)     = Q_GT;
               
%        showImage8_zoomin(I_GT,printEPS,2,flag_cut_bounds,dim_cut,thvalues,L, location1, location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_gt', '.eps'))
    end
    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% CS-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
    %% ====== 1) BT-H Method ======
    if ismember('BT-H',algorithms)
        alg = alg + 1;
        
        cd BT-H
        t2=tic;
        I_BT_H = BroveyRegHazeMin(I_MS,I_PAN,ratio);
        time_BT_H = toc(t2);
        fprintf('Elaboration time BT-H: %.2f [sec]\n',time_BT_H);
        cd ..

        %%% Quality indexes computation                           
        [Q_avg_BT_H, SAM_BT_H, ERGAS_BT_H, SCC_BT_H, Q_BT_H] = indexes_evaluation(I_BT_H,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_BT_H,Q_avg_BT_H,SAM_BT_H,ERGAS_BT_H,SCC_BT_H];
        MatrixImage(:,:,:,alg) = I_BT_H;
        
        Q_avg_BT_H_multiexm(i) = Q_avg_BT_H;
        SAM_BT_H_multiexm(i)   = SAM_BT_H;
        ERGAS_BT_H_multiexm(i) = ERGAS_BT_H;
        SCC_BT_H_multiexm(i)   = SCC_BT_H;
        Q_BT_H_multiexm(i)     = Q_BT_H;
        
%        showImage8_zoomin(I_BT_H,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_bth.eps'))
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

        [Q_avg_BDSD_PC, SAM_BDSD_PC, ERGAS_BDSD_PC, SCC_BDSD_PC, Q_BDSD_PC] = indexes_evaluation(I_BDSD_PC,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_BDSD_PC,Q_avg_BDSD_PC,SAM_BDSD_PC,ERGAS_BDSD_PC,SCC_BDSD_PC];
        MatrixImage(:,:,:,alg) = I_BDSD_PC;
        
        Q_avg_BDSD_PC_multiexm(i) = Q_avg_BDSD_PC;
        SAM_BDSD_PC_multiexm(i)   = SAM_BDSD_PC;
        ERGAS_BDSD_PC_multiexm(i) = ERGAS_BDSD_PC;
        SCC_BDSD_PC_multiexm(i)   = SCC_BDSD_PC;
        Q_BDSD_PC_multiexm(i)     = Q_BDSD_PC;
        
%        showImage8_zoomin(I_BDSD_PC,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_bdsd_pc.eps'))
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% MRA-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%    
  
     %% ====== 1) MTF-GLP-HPM-R Method ======  
    if ismember('MTF-GLP-HPM-R',algorithms)
        alg = alg + 1;

        cd GLP
        t2=tic;
        I_MTF_GLP_HPM_R = MTF_GLP_HPM_R(I_MS,I_PAN,sensor,ratio);
        time_MTF_GLP_HPM_R = toc(t2);
        fprintf('Elaboration time MTF-GLP: %.2f [sec]\n',time_MTF_GLP_HPM_R);
        cd ..

        [Q_avg_MTF_GLP_HPM_R, SAM_MTF_GLP_HPM_R, ERGAS_MTF_GLP_HPM_R, SCC_MTF_GLP_HPM_R, Q_MTF_GLP_HPM_R] = indexes_evaluation(I_MTF_GLP_HPM_R,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_MTF_GLP_HPM_R,Q_avg_MTF_GLP_HPM_R,SAM_MTF_GLP_HPM_R,ERGAS_MTF_GLP_HPM_R,SCC_MTF_GLP_HPM_R];
        MatrixImage(:,:,:,alg) = I_MTF_GLP_HPM_R;
        
        Q_avg_MTF_GLP_HPM_R_multiexm(i) = Q_avg_MTF_GLP_HPM_R;
        SAM_MTF_GLP_HPM_R_multiexm(i)   = SAM_MTF_GLP_HPM_R;
        ERGAS_MTF_GLP_HPM_R_multiexm(i) = ERGAS_MTF_GLP_HPM_R;
        SCC_MTF_GLP_HPM_R_multiexm(i)   = SCC_MTF_GLP_HPM_R;
        Q_MTF_GLP_HPM_R_multiexm(i)     = Q_MTF_GLP_HPM_R;
        
%        showImage8_zoomin(I_MTF_GLP_HPM_R,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_mtfglp_hpm_r.eps'))

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

        [Q_avg_MTF_GLP_FS, SAM_MTF_GLP_FS, ERGAS_MTF_GLP_FS, SCC_MTF_GLP_FS, Q_MTF_GLP_FS] = indexes_evaluation(I_MTF_GLP_FS,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_MTF_GLP_FS,Q_avg_MTF_GLP_FS,SAM_MTF_GLP_FS,ERGAS_MTF_GLP_FS,SCC_MTF_GLP_FS];
        MatrixImage(:,:,:,alg) = I_MTF_GLP_FS;
        
        Q_avg_MTF_GLP_FS_multiexm(i) = Q_avg_MTF_GLP_FS;
        SAM_MTF_GLP_FS_multiexm(i)   = SAM_MTF_GLP_FS;
        ERGAS_MTF_GLP_FS_multiexm(i) = ERGAS_MTF_GLP_FS;
        SCC_MTF_GLP_FS_multiexm(i)   = SCC_MTF_GLP_FS;
        Q_MTF_GLP_FS_multiexm(i)     = Q_MTF_GLP_FS;        
        
%        showImage8_zoomin(I_MTF_GLP_FS,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_mtfglpfs.eps'))
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

        [Q_avg_TV, SAM_TV, ERGAS_TV, SCC_TV, Q_TV] = indexes_evaluation(I_TV,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
        MatrixResults(alg,:) = [Q_TV,Q_avg_TV,SAM_TV,ERGAS_TV,SCC_TV];
        MatrixImage(:,:,:,alg) = I_TV;

        Q_avg_TV_multiexm(i) = Q_avg_TV;
        SAM_TV_multiexm(i)   = SAM_TV;
        ERGAS_TV_multiexm(i) = ERGAS_TV;
        SCC_TV_multiexm(i)   = SCC_TV;
        Q_TV_multiexm(i)     = Q_TV;  
        
%        showImage8_zoomin(I_TV,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1), '_tv.eps'))
    end    

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% DL-based Methods %%%%%%%%%%%%%%%%%%%%%%%%%%
%% ====== 1) PNN Method ======
 
%    load(strcat('2_DL_Result/WV3_Reduced/PNN/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for PNN
%    I_pnn = double(sr);
%
%    if ismember('PNN',algorithms)
%        alg = alg + 1;
%        [Q_avg_pnn, SAM_pnn, ERGAS_pnn, SCC_pnn, Q_pnn] = indexes_evaluation(I_pnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_pnn,Q_avg_pnn,SAM_pnn,ERGAS_pnn,SCC_pnn];
%        MatrixImage(:,:,:,alg) = I_pnn;
%
%        Q_avg_pnn_multiexm(i) = Q_avg_pnn;
%        SAM_pnn_multiexm(i)   = SAM_pnn;
%        ERGAS_pnn_multiexm(i) = ERGAS_pnn;
%        SCC_pnn_multiexm(i)   = SCC_pnn;
%        Q_pnn_multiexm(i)     = Q_pnn;
%
%        showImage8_zoomin(I_pnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_pnn.eps'))
%    end
%    %% ====== 2) PanNet Method ======
%
%    load(strcat('2_DL_Result/WV3_Reduced/PanNet/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for PanNet
%    I_pannet = double(sr);
%
%    if ismember('PanNet',algorithms)
%        alg = alg + 1;
%        [Q_avg_pannet, SAM_pannet, ERGAS_pannet, SCC_pannet, Q_pannet] = indexes_evaluation(I_pannet,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_pannet,Q_avg_pannet,SAM_pannet,ERGAS_pannet,SCC_pannet];
%        MatrixImage(:,:,:,alg) = I_pannet;
%
%
%        Q_avg_pannet_multiexm(i) = Q_avg_pannet;
%        SAM_pannet_multiexm(i)   = SAM_pannet;
%        ERGAS_pannet_multiexm(i) = ERGAS_pannet;
%        SCC_pannet_multiexm(i)   = SCC_pannet;
%        Q_pannet_multiexm(i)     = Q_pannet;
%
%        showImage8_zoomin(I_pannet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_pannet.eps'))
%    end
    
    
    

%% ====== 3) DiCNN Method ======
%    load(strcat('2_DL_Result/WV3_Reduced/DiCNN/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for DiCNN
%    I_dicnn = double(sr);
%
%    if ismember('DiCNN',algorithms)
%        alg = alg + 1;
%        [Q_avg_dicnn, SAM_dicnn, ERGAS_dicnn, SCC_dicnn, Q_dicnn] = indexes_evaluation(I_dicnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_dicnn,Q_avg_dicnn,SAM_dicnn,ERGAS_dicnn,SCC_dicnn];
%        MatrixImage(:,:,:,alg) = I_dicnn;
%
%        Q_avg_dicnn_multiexm(i) = Q_avg_dicnn;
%        SAM_dicnn_multiexm(i)   = SAM_dicnn;
%        ERGAS_dicnn_multiexm(i) = ERGAS_dicnn;
%        SCC_dicnn_multiexm(i)   = SCC_dicnn;
%        Q_dicnn_multiexm(i)     = Q_dicnn;
%
%        showImage8_zoomin(I_dicnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_dicnn.eps'))
%    end
%
%    %% ====== 4) MSDCNN Method ======
%    load(strcat('2_DL_Result/WV3_Reduced/MSDCNN/results/output_mulExm_',  num2str(i-1), '.mat')) % load i-th image for MSDCNN
%    I_msdcnn = double(sr);
%
%    if ismember('MSDCNN',algorithms)
%        alg = alg + 1;
%        [Q_avg_msdcnn, SAM_msdcnn, ERGAS_msdcnn, SCC_msdcnn, Q_msdcnn] = indexes_evaluation(I_msdcnn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_msdcnn,Q_avg_msdcnn,SAM_msdcnn,ERGAS_msdcnn,SCC_msdcnn];
%        MatrixImage(:,:,:,alg) = I_msdcnn;
%
%        Q_avg_msdcnn_multiexm(i) = Q_avg_msdcnn;
%        SAM_msdcnn_multiexm(i)   = SAM_msdcnn;
%        ERGAS_msdcnn_multiexm(i) = ERGAS_msdcnn;
%        SCC_msdcnn_multiexm(i)   = SCC_msdcnn;
%        Q_msdcnn_multiexm(i)     = Q_msdcnn;
%
%        showImage8_zoomin(I_msdcnn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_msdcnn.eps'))
%    end
%    %% ====== 5) BDPN Method ======
%    load(strcat('2_DL_Result/WV3_Reduced/BDPN/results/output_mulExm_' , num2str(i-1), '.mat')) % load i-th image for BDPN
%    I_bdpn  = double(sr);
%
%    if ismember('BDPN',algorithms)
%        alg = alg + 1;
%        [Q_avg_bdpn, SAM_bdpn, ERGAS_bdpn, SCC_bdpn, Q_bdpn] = indexes_evaluation(I_bdpn,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_bdpn,Q_avg_bdpn,SAM_bdpn,ERGAS_bdpn,SCC_bdpn];
%        MatrixImage(:,:,:,alg) = I_bdpn;
%
%        Q_avg_bdpn_multiexm(i) = Q_avg_bdpn;
%        SAM_bdpn_multiexm(i)   = SAM_bdpn;
%        ERGAS_bdpn_multiexm(i) = ERGAS_bdpn;
%        SCC_bdpn_multiexm(i)   = SCC_bdpn;
%        Q_bdpn_multiexm(i)     = Q_bdpn;
%
%        showImage8_zoomin(I_bdpn,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_bdpn.eps'))
%    end
%
%     %% ====== 6) FusionNet Method ======
%    load(strcat('2_DL_Result/WV3_Reduced/FusionNet/results/output_mulExm_', num2str(i-1), '.mat')) % load i-th image for FusionNet
%    I_fusionnet = double(sr);
%
%    if ismember('FusionNet',algorithms)
%        alg = alg + 1;
%        [Q_avg_fusionnet, SAM_fusionnet, ERGAS_fusionnet, SCC_fusionnet, Q_fusionnet] = indexes_evaluation(I_fusionnet,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_fusionnet,Q_avg_fusionnet,SAM_fusionnet,ERGAS_fusionnet,SCC_fusionnet];
%        MatrixImage(:,:,:,alg) = I_fusionnet;
%
%        Q_avg_fusionnet_multiexm(i) = Q_avg_fusionnet;
%        SAM_fusionnet_multiexm(i)   = SAM_fusionnet;
%        ERGAS_fusionnet_multiexm(i) = ERGAS_fusionnet;
%        SCC_fusionnet_multiexm(i)   = SCC_fusionnet;
%        Q_fusionnet_multiexm(i)     = Q_fusionnet;
%
%        showImage8_zoomin(I_fusionnet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_fusionnet.eps'))
%    end
   
        %% ====== 7) LAGConv Method ======
%    load(strcat('2_DL_Result/WV3_Reduced/LAGConv/results/output_mulExm_',num2str(i-1), '.mat')) % load i-th image for LAGConv
%    I_lagnet = double(sr);
%
%    if ismember('LAGConv',algorithms)
%        alg = alg + 1;
%        [Q_avg_lagnet, SAM_lagnet, ERGAS_lagnet, SCC_lagnet, Q_lagnet] = indexes_evaluation(I_lagnet,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
%        MatrixResults(alg,:) = [Q_lagnet,Q_avg_lagnet,SAM_lagnet,ERGAS_lagnet,SCC_lagnet];
%        MatrixImage(:,:,:,alg) = I_lagnet;
%
%        Q_avg_lagnet_multiexm(i) = Q_avg_lagnet;
%        SAM_lagnet_multiexm(i)   = SAM_lagnet;
%        ERGAS_lagnet_multiexm(i) = ERGAS_lagnet;
%        SCC_lagnet_multiexm(i)   = SCC_lagnet;
%        Q_lagnet_multiexm(i)     = Q_lagnet;
%
%        showImage8_zoomin(I_lagnet,printEPS,1,flag_cut_bounds,dim_cut,thvalues,L,location1,location2);
%        pause(2);print('-depsc', strcat(data_name, num2str(i-1),'_lagnet.eps'))
%    end
   
    
    
end

%% Print in LATEX

%matrix2latex(MatrixResults(:,[1,3,4]),'RR_Assessment.tex', 'rowLabels',algorithms,'columnLabels',[{'Q2n'},{'SAM'},{'ERGAS'}],'alignment','c','format', '%.4f');

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
avg_Q_GT_multiexm = mean(Q_GT_multiexm);
std_Q_GT_multiexm = std(Q_GT_multiexm);

avg_Q_avg_GT_multiexm = mean(Q_avg_GT_multiexm);
std_Q_avg_GT_multiexm = std(Q_avg_GT_multiexm);

avg_SAM_GT_multiexm = mean(SAM_GT_multiexm);
std_SAM_GT_multiexm = std(SAM_GT_multiexm);

avg_ERGAS_GT_multiexm = mean(ERGAS_GT_multiexm);
std_ERGAS_GT_multiexm = std(ERGAS_GT_multiexm);

avg_SCC_GT_multiexm = mean(SCC_GT_multiexm);
std_SCC_GT_multiexm = std(SCC_GT_multiexm);

Avg_MatrixResults(1,:) = [avg_Q_GT_multiexm, std_Q_GT_multiexm, avg_Q_avg_GT_multiexm, std_Q_avg_GT_multiexm, ...
                          avg_SAM_GT_multiexm, std_SAM_GT_multiexm, avg_ERGAS_GT_multiexm, std_ERGAS_GT_multiexm,...
                          avg_SCC_GT_multiexm, std_SCC_GT_multiexm];
               
 
                     
% BT_H: average Q_avg
% avg_Q_BT_H_multiexm = mean(Q_BT_H_multiexm);
% std_Q_BT_H_multiexm = std(Q_BT_H_multiexm);

% avg_Q_avg_BT_H_multiexm = mean(Q_avg_BT_H_multiexm);
% std_Q_avg_BT_H_multiexm = std(Q_avg_BT_H_multiexm);

% avg_SAM_BT_H_multiexm = mean(SAM_BT_H_multiexm);
% std_SAM_BT_H_multiexm = std(SAM_BT_H_multiexm);

% avg_ERGAS_BT_H_multiexm = mean(ERGAS_BT_H_multiexm);
% std_ERGAS_BT_H_multiexm = std(ERGAS_BT_H_multiexm);

% avg_SCC_BT_H_multiexm = mean(SCC_BT_H_multiexm);
% std_SCC_BT_H_multiexm = std(SCC_BT_H_multiexm);

% Avg_MatrixResults(2,:) = [avg_Q_BT_H_multiexm, std_Q_BT_H_multiexm, avg_Q_avg_BT_H_multiexm, std_Q_avg_BT_H_multiexm, ...
%                           avg_SAM_BT_H_multiexm, std_SAM_BT_H_multiexm, avg_ERGAS_BT_H_multiexm, std_ERGAS_BT_H_multiexm,...
%                           avg_SCC_BT_H_multiexm, std_SCC_BT_H_multiexm];                      
                      
% % BDSD: average Q_avg
% avg_Q_BDSD_PC_multiexm = mean(Q_BDSD_PC_multiexm);
% std_Q_BDSD_PC_multiexm = std(Q_BDSD_PC_multiexm);

% avg_Q_avg_BDSD_PC_multiexm = mean(Q_avg_BDSD_PC_multiexm);
% std_Q_avg_BDSD_PC_multiexm = std(Q_avg_BDSD_PC_multiexm);

% avg_SAM_BDSD_PC_multiexm = mean(SAM_BDSD_PC_multiexm);
% std_SAM_BDSD_PC_multiexm = std(SAM_BDSD_PC_multiexm);

% avg_ERGAS_BDSD_PC_multiexm = mean(ERGAS_BDSD_PC_multiexm);
% std_ERGAS_BDSD_PC_multiexm = std(ERGAS_BDSD_PC_multiexm);

% avg_SCC_BDSD_PC_multiexm = mean(SCC_BDSD_PC_multiexm);
% std_SCC_BDSD_PC_multiexm = std(SCC_BDSD_PC_multiexm);

% Avg_MatrixResults(3,:) = [avg_Q_BDSD_PC_multiexm, std_Q_BDSD_PC_multiexm, avg_Q_avg_BDSD_PC_multiexm, std_Q_avg_BDSD_PC_multiexm, ...
%                           avg_SAM_BDSD_PC_multiexm, std_SAM_BDSD_PC_multiexm, avg_ERGAS_BDSD_PC_multiexm, std_ERGAS_BDSD_PC_multiexm,...
%                           avg_SCC_BDSD_PC_multiexm, std_SCC_BDSD_PC_multiexm];   
                   

                                           
% % MTF_GLP: average Q_avg
% avg_Q_MTF_GLP_HPM_R_multiexm = mean(Q_MTF_GLP_HPM_R_multiexm);
% std_Q_MTF_GLP_HPM_R_multiexm = std(Q_MTF_GLP_HPM_R_multiexm);

% avg_Q_avg_MTF_GLP_HPM_R_multiexm = mean(Q_avg_MTF_GLP_HPM_R_multiexm);
% std_Q_avg_MTF_GLP_HPM_R_multiexm = std(Q_avg_MTF_GLP_HPM_R_multiexm);

% avg_SAM_MTF_GLP_HPM_R_multiexm = mean(SAM_MTF_GLP_HPM_R_multiexm);
% std_SAM_MTF_GLP_HPM_R_multiexm = std(SAM_MTF_GLP_HPM_R_multiexm);

% avg_ERGAS_MTF_GLP_HPM_R_multiexm = mean(ERGAS_MTF_GLP_HPM_R_multiexm);
% std_ERGAS_MTF_GLP_HPM_R_multiexm = std(ERGAS_MTF_GLP_HPM_R_multiexm);

% avg_SCC_MTF_GLP_HPM_R_multiexm = mean(SCC_MTF_GLP_HPM_R_multiexm);
% std_SCC_MTF_GLP_HPM_R_multiexm = std(SCC_MTF_GLP_HPM_R_multiexm);

% Avg_MatrixResults(4,:) = [avg_Q_MTF_GLP_HPM_R_multiexm, std_Q_MTF_GLP_HPM_R_multiexm, avg_Q_avg_MTF_GLP_HPM_R_multiexm, std_Q_avg_MTF_GLP_HPM_R_multiexm, ...
%                           avg_SAM_MTF_GLP_HPM_R_multiexm, std_SAM_MTF_GLP_HPM_R_multiexm, avg_ERGAS_MTF_GLP_HPM_R_multiexm, std_ERGAS_MTF_GLP_HPM_R_multiexm,...
%                           avg_SCC_MTF_GLP_HPM_R_multiexm, std_SCC_MTF_GLP_HPM_R_multiexm];                        
                      
                      
                      
% % MTF_GLP_FS: average Q_avg
% avg_Q_MTF_GLP_FS_multiexm = mean(Q_MTF_GLP_FS_multiexm);
% std_Q_MTF_GLP_FS_multiexm = std(Q_MTF_GLP_FS_multiexm);

% avg_Q_avg_MTF_GLP_FS_multiexm = mean(Q_avg_MTF_GLP_FS_multiexm);
% std_Q_avg_MTF_GLP_FS_multiexm = std(Q_avg_MTF_GLP_FS_multiexm);

% avg_SAM_MTF_GLP_FS_multiexm = mean(SAM_MTF_GLP_FS_multiexm);
% std_SAM_MTF_GLP_FS_multiexm = std(SAM_MTF_GLP_FS_multiexm);

% avg_ERGAS_MTF_GLP_FS_multiexm = mean(ERGAS_MTF_GLP_FS_multiexm);
% std_ERGAS_MTF_GLP_FS_multiexm = std(ERGAS_MTF_GLP_FS_multiexm);

% avg_SCC_MTF_GLP_FS_multiexm = mean(SCC_MTF_GLP_FS_multiexm);
% std_SCC_MTF_GLP_FS_multiexm = std(SCC_MTF_GLP_FS_multiexm);

% Avg_MatrixResults(5,:) = [avg_Q_MTF_GLP_FS_multiexm, std_Q_MTF_GLP_FS_multiexm, avg_Q_avg_MTF_GLP_FS_multiexm, std_Q_avg_MTF_GLP_FS_multiexm, ...
%                           avg_SAM_MTF_GLP_FS_multiexm, std_SAM_MTF_GLP_FS_multiexm, avg_ERGAS_MTF_GLP_FS_multiexm, std_ERGAS_MTF_GLP_FS_multiexm,...
%                           avg_SCC_MTF_GLP_FS_multiexm, std_SCC_MTF_GLP_FS_multiexm];                         
                      
                      
% % TV: average Q_avg
% avg_Q_TV_multiexm = mean(Q_TV_multiexm);
% std_Q_TV_multiexm = std(Q_TV_multiexm);

% avg_Q_avg_TV_multiexm = mean(Q_avg_TV_multiexm);
% std_Q_avg_TV_multiexm = std(Q_avg_TV_multiexm);

% avg_SAM_TV_multiexm = mean(SAM_TV_multiexm);
% std_SAM_TV_multiexm = std(SAM_TV_multiexm);

% avg_ERGAS_TV_multiexm = mean(ERGAS_TV_multiexm);
% std_ERGAS_TV_multiexm = std(ERGAS_TV_multiexm);

% avg_SCC_TV_multiexm = mean(SCC_TV_multiexm);
% std_SCC_TV_multiexm = std(SCC_TV_multiexm);

% Avg_MatrixResults(6,:) = [avg_Q_TV_multiexm, std_Q_TV_multiexm, avg_Q_avg_TV_multiexm, std_Q_avg_TV_multiexm, ...
%                           avg_SAM_TV_multiexm, std_SAM_TV_multiexm, avg_ERGAS_TV_multiexm, std_ERGAS_TV_multiexm,...
%                           avg_SCC_TV_multiexm, std_SCC_TV_multiexm];   
% pnn: average Q_avg
%avg_Q_pnn_multiexm = mean(Q_pnn_multiexm);
%std_Q_pnn_multiexm = std(Q_pnn_multiexm);
%
%avg_Q_avg_pnn_multiexm = mean(Q_avg_pnn_multiexm);
%std_Q_avg_pnn_multiexm = std(Q_avg_pnn_multiexm);
%
%avg_SAM_pnn_multiexm = mean(SAM_pnn_multiexm);
%std_SAM_pnn_multiexm = std(SAM_pnn_multiexm);
%
%avg_ERGAS_pnn_multiexm = mean(ERGAS_pnn_multiexm);
%std_ERGAS_pnn_multiexm = std(ERGAS_pnn_multiexm);
%
%avg_SCC_pnn_multiexm = mean(SCC_pnn_multiexm);
%std_SCC_pnn_multiexm = std(SCC_pnn_multiexm);
%
%Avg_MatrixResults(7,:) = [avg_Q_pnn_multiexm, std_Q_pnn_multiexm, avg_Q_avg_pnn_multiexm, std_Q_avg_pnn_multiexm, ...
%                          avg_SAM_pnn_multiexm, std_SAM_pnn_multiexm, avg_ERGAS_pnn_multiexm, std_ERGAS_pnn_multiexm,...
%                          avg_SCC_pnn_multiexm, std_SCC_pnn_multiexm];
% pannet: average Q_avg
%avg_Q_pannet_multiexm = mean(Q_pannet_multiexm);
%std_Q_pannet_multiexm = std(Q_pannet_multiexm);
%
%avg_Q_avg_pannet_multiexm = mean(Q_avg_pannet_multiexm);
%std_Q_avg_pannet_multiexm = std(Q_avg_pannet_multiexm);
%
%avg_SAM_pannet_multiexm = mean(SAM_pannet_multiexm);
%std_SAM_pannet_multiexm = std(SAM_pannet_multiexm);
%
%avg_ERGAS_pannet_multiexm = mean(ERGAS_pannet_multiexm);
%std_ERGAS_pannet_multiexm = std(ERGAS_pannet_multiexm);
%
%avg_SCC_pannet_multiexm = mean(SCC_pannet_multiexm);
%std_SCC_pannet_multiexm = std(SCC_pannet_multiexm);
%
%Avg_MatrixResults(8,:) = [avg_Q_pannet_multiexm, std_Q_pannet_multiexm, avg_Q_avg_pannet_multiexm, std_Q_avg_pannet_multiexm, ...
%                          avg_SAM_pannet_multiexm, std_SAM_pannet_multiexm, avg_ERGAS_pannet_multiexm, std_ERGAS_pannet_multiexm,...
%                          avg_SCC_pannet_multiexm, std_SCC_pannet_multiexm];
%
%% dicnn: average Q_avg
%avg_Q_dicnn_multiexm = mean(Q_dicnn_multiexm);
%std_Q_dicnn_multiexm = std(Q_dicnn_multiexm);
%
%avg_Q_avg_dicnn_multiexm = mean(Q_avg_dicnn_multiexm);
%std_Q_avg_dicnn_multiexm = std(Q_avg_dicnn_multiexm);
%
%avg_SAM_dicnn_multiexm = mean(SAM_dicnn_multiexm);
%std_SAM_dicnn_multiexm = std(SAM_dicnn_multiexm);
%
%avg_ERGAS_dicnn_multiexm = mean(ERGAS_dicnn_multiexm);
%std_ERGAS_dicnn_multiexm = std(ERGAS_dicnn_multiexm);
%
%avg_SCC_dicnn_multiexm = mean(SCC_dicnn_multiexm);
%std_SCC_dicnn_multiexm = std(SCC_dicnn_multiexm);
%
%Avg_MatrixResults(9,:) = [avg_Q_dicnn_multiexm, std_Q_dicnn_multiexm, avg_Q_avg_dicnn_multiexm, std_Q_avg_dicnn_multiexm, ...
%                          avg_SAM_dicnn_multiexm, std_SAM_dicnn_multiexm, avg_ERGAS_dicnn_multiexm, std_ERGAS_dicnn_multiexm,...
%                          avg_SCC_dicnn_multiexm, std_SCC_dicnn_multiexm];
%
%
%
%%
%% msdcnn: average Q_avg
%avg_Q_msdcnn_multiexm = mean(Q_msdcnn_multiexm);
%std_Q_msdcnn_multiexm = std(Q_msdcnn_multiexm);
%
%avg_Q_avg_msdcnn_multiexm = mean(Q_avg_msdcnn_multiexm);
%std_Q_avg_msdcnn_multiexm = std(Q_avg_msdcnn_multiexm);
%
%avg_SAM_msdcnn_multiexm = mean(SAM_msdcnn_multiexm);
%std_SAM_msdcnn_multiexm = std(SAM_msdcnn_multiexm);
%
%avg_ERGAS_msdcnn_multiexm = mean(ERGAS_msdcnn_multiexm);
%std_ERGAS_msdcnn_multiexm = std(ERGAS_msdcnn_multiexm);
%
%avg_SCC_msdcnn_multiexm = mean(SCC_msdcnn_multiexm);
%std_SCC_msdcnn_multiexm = std(SCC_msdcnn_multiexm);
%
%Avg_MatrixResults(10,:) = [avg_Q_msdcnn_multiexm, std_Q_msdcnn_multiexm, avg_Q_avg_msdcnn_multiexm, std_Q_avg_msdcnn_multiexm, ...
%                          avg_SAM_msdcnn_multiexm, std_SAM_msdcnn_multiexm, avg_ERGAS_msdcnn_multiexm, std_ERGAS_msdcnn_multiexm,...
%                          avg_SCC_msdcnn_multiexm, std_SCC_msdcnn_multiexm];
%
%
%% bdpn: average Q_avg
%avg_Q_bdpn_multiexm = mean(Q_bdpn_multiexm);
%std_Q_bdpn_multiexm = std(Q_bdpn_multiexm);
%
%avg_Q_avg_bdpn_multiexm = mean(Q_avg_bdpn_multiexm);
%std_Q_avg_bdpn_multiexm = std(Q_avg_bdpn_multiexm);
%
%avg_SAM_bdpn_multiexm = mean(SAM_bdpn_multiexm);
%std_SAM_bdpn_multiexm = std(SAM_bdpn_multiexm);
%
%avg_ERGAS_bdpn_multiexm = mean(ERGAS_bdpn_multiexm);
%std_ERGAS_bdpn_multiexm = std(ERGAS_bdpn_multiexm);
%
%avg_SCC_bdpn_multiexm = mean(SCC_bdpn_multiexm);
%std_SCC_bdpn_multiexm = std(SCC_bdpn_multiexm);
%
%Avg_MatrixResults(11,:) = [avg_Q_bdpn_multiexm, std_Q_bdpn_multiexm, avg_Q_avg_bdpn_multiexm, std_Q_avg_bdpn_multiexm, ...
%                          avg_SAM_bdpn_multiexm, std_SAM_bdpn_multiexm, avg_ERGAS_bdpn_multiexm, std_ERGAS_bdpn_multiexm,...
%                          avg_SCC_bdpn_multiexm, std_SCC_bdpn_multiexm];
%
%
%
%% fusionnet: average Q_avg
%avg_Q_fusionnet_multiexm = mean(Q_fusionnet_multiexm);
%std_Q_fusionnet_multiexm = std(Q_fusionnet_multiexm);
%
%avg_Q_avg_fusionnet_multiexm = mean(Q_avg_fusionnet_multiexm);
%std_Q_avg_fusionnet_multiexm = std(Q_avg_fusionnet_multiexm);
%
%avg_SAM_fusionnet_multiexm = mean(SAM_fusionnet_multiexm);
%std_SAM_fusionnet_multiexm = std(SAM_fusionnet_multiexm);
%
%avg_ERGAS_fusionnet_multiexm = mean(ERGAS_fusionnet_multiexm);
%std_ERGAS_fusionnet_multiexm = std(ERGAS_fusionnet_multiexm);
%
%avg_SCC_fusionnet_multiexm = mean(SCC_fusionnet_multiexm);
%std_SCC_fusionnet_multiexm = std(SCC_fusionnet_multiexm);
%
%Avg_MatrixResults(12,:) = [avg_Q_fusionnet_multiexm, std_Q_fusionnet_multiexm, avg_Q_avg_fusionnet_multiexm, std_Q_avg_fusionnet_multiexm, ...
%                          avg_SAM_fusionnet_multiexm, std_SAM_fusionnet_multiexm, avg_ERGAS_fusionnet_multiexm, std_ERGAS_fusionnet_multiexm,...
%                          avg_SCC_fusionnet_multiexm, std_SCC_fusionnet_multiexm];
%% lagnet: average Q_avg
%avg_Q_lagnet_multiexm = mean(Q_lagnet_multiexm);
%std_Q_lagnet_multiexm = std(Q_lagnet_multiexm);
%
%avg_Q_avg_lagnet_multiexm = mean(Q_avg_lagnet_multiexm);
%std_Q_avg_lagnet_multiexm = std(Q_avg_lagnet_multiexm);
%
%avg_SAM_lagnet_multiexm = mean(SAM_lagnet_multiexm);
%std_SAM_lagnet_multiexm = std(SAM_lagnet_multiexm);
%
%avg_ERGAS_lagnet_multiexm = mean(ERGAS_lagnet_multiexm);
%std_ERGAS_lagnet_multiexm = std(ERGAS_lagnet_multiexm);
%
%avg_SCC_lagnet_multiexm = mean(SCC_lagnet_multiexm);
%std_SCC_lagnet_multiexm = std(SCC_lagnet_multiexm);
%
%Avg_MatrixResults(13,:) = [avg_Q_lagnet_multiexm, std_Q_lagnet_multiexm, avg_Q_avg_lagnet_multiexm, std_Q_avg_lagnet_multiexm, ...
%                          avg_SAM_lagnet_multiexm, std_SAM_lagnet_multiexm, avg_ERGAS_lagnet_multiexm, std_ERGAS_lagnet_multiexm,...
%                          avg_SCC_lagnet_multiexm, std_SCC_lagnet_multiexm];
                                            
              
                      
%matrix2latex(Avg_MatrixResults(:,[1,2, 5,6, 7,8 ]),'Avg_RR_Assessment.tex', 'rowLabels',algorithms,'columnLabels',[{'Q2n'}, {'Q2n-std'}, {'SAM'}, {'SAM-std'}, {'ERGAS'}, {'ERGAS-std'}],'alignment','c','format', '%.4f');

fprintf('\n')
disp('#######################################################')
%disp(['Display the performance for:', num2str(1:i)])
disp('#######################################################')
disp(' |====Q====|===Q_avg===|=====SAM=====|======ERGAS=======|=======SCC=======')
Avg_MatrixResults


