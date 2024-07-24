function res = analysis_ref_batched_images(path, ratio, full_res, const)
    % data should be [0, max_range]
    % e.g. wv3: max_range: 2047
    data = load(path);
    gt = data.gt;
    sr = data.sr;
    addpath('./Tools')
    addpath('./Quality_Indices/')
    Q_block_size = 32;
    thvalues = 0;
    
    L = 11;
    flag_cut_bounds = 0;

    if full_res
        dim_cut = 21;
    else
        dim_cut = 30;
    end

    bs = size(sr, 1);
    res = {};

    sam = zeros(1, bs);
    ergas = zeros(1, bs);
    scc = zeros(1, bs);
    qn = zeros(1, bs);
    qave = zeros(1, bs);
    psnr = zeros(1, bs);
    ssim = zeros(1, bs);

    for i = (1:bs)
        sr1 = permute(squeeze(sr(i, :, :, :)), [2, 3, 1]);
        gt1 = permute(squeeze(gt(i, :, :, :)), [2, 3, 1]);
        [q_avg_gt, sam_gt, ergas_gt, scc_gt, q_gt] = indexes_evaluation(sr1, gt1, ratio, L, Q_block_size, flag_cut_bounds, dim_cut, thvalues);
        [psnr_gt, ssim_gt] = quality_assess(sr1 / const, gt1 / const);

        sam(i) = sam_gt;
        ergas(i) = ergas_gt;
        scc(i) = scc_gt;
        qn(i) = q_gt;
        qave(i) = q_avg_gt;
        psnr(i) = psnr_gt;
        ssim(i) = ssim_gt;
        fprintf("sample %d - sam: %f, ergas: %f, scc: %f, qn: %f, q_ave: %f, psnr: %f, ssim: %f \n", i, sam_gt, ergas_gt, scc_gt, q_gt, q_avg_gt, psnr_gt, ssim_gt)
    end

    res.sam = [mean(sam), std(sam)];
    res.ergas = [mean(ergas), std(ergas)];
    res.scc = [mean(scc), std(scc)];
    res.qn = [mean(qn), std(qn)];
    res.qave = [mean(qave), std(qave)];
    res.psnr = [mean(psnr), std(psnr)];
    res.ssim = [mean(ssim), std(ssim)];

end
