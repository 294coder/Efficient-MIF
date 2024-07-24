function res = analysis_ref_unbatched_images(dir_path, ratio, sensor, flag_cut_bounds, printout)
    % data should be [0, max_range]
    % e.g. wv3: max_range: 2047

    if nargin == 3
        flag_cut_bounds = 0;
        printout = 1;
    elseif nargin == 4
        printout = 1;
    end

    sensor = lower(sensor);

    if sensor == "wv3" || sensor == "wv2" || sensor == "qb"
        const = 2047;
        gt_key = 'gt';

        if sensor == "wv3"
            C = 8;
            gt_path = "/Data2/ZiHanCao/datasets/pansharpening/wv3/reduced_examples/test_wv3_multiExm1.h5";
        elseif sensor == "wv2"
            C = 8;
            gt_path = '/Data2/ZiHanCao/datasets/pansharpening/wv2/reduced_examples/test_wv2_multiExm1.h5';
        else
            C = 4;
            gt_path = "/Data2/ZiHanCao/datasets/pansharpening/qb/reduced_examples/test_qb_multiExm1.h5";
        end

    elseif sensor == "gf2" || sensor == "gf"
        const = 1023;
        gt_key = 'gt';
        C = 4;
        gt_path = "/Data2/ZiHanCao/datasets/pansharpening/gf/reduced_examples/test_gf2_multiExm1.h5";
    elseif sensor == "cave" || sensor == "harvard"
        const = 1;
        gt_key = 'GT';
        C = 31;

        if sensor == "cave"

            if ratio == 4
                gt_path = "/Data2/ZiHanCao/datasets/HISI/new_cave/test_cave(with_up)x4.h5";
            elseif ratio == 8
                gt_path = "/Data2/ZiHanCao/datasets/HISI/new_cave/x8/test_cave(with_up)x8_rgb.h5";
            end

        else % % harvard
            gt_path = '';
        end

    else
        error(strcat(sensor, ' is not supported!'))

    end

    gts = h5read(gt_path, strcat('/', gt_key));
    gts = permute(gts, [4, 2, 1, 3]);
    bs = size(gts, 1);

    addpath('./Tools')
    addpath('./Quality_Indices/')
    Q_block_size = 32;
    thvalues = 0;
    L = 11;
    full_res = 0; % default to 0

    if full_res
        dim_cut = 21;
    else
        dim_cut = 30;
    end

    res = {};

    sam = zeros(1, bs);
    ergas = zeros(1, bs);
    scc = zeros(1, bs);
    qn = zeros(1, bs);
    qave = zeros(1, bs);
    psnr = zeros(1, bs);
    ssim = zeros(1, bs);

    for i = (0:bs - 1)
        p = strcat(dir_path, "/", "output_mulExm_", string(i), ".mat");
        data = load(p);
        sr_key = fieldnames(data);
        sr_key = sr_key{1};
        sr1 = data.(string(sr_key));

        % check the size
        if size(sr1, 1) == 1 && length(size(sr1)) == 4
            sr1 = squeeze(sr1);
        end

        if size(sr1, 3) ~= C && length(size(sr1)) == 3
            sr1 = permute(sr1, [2, 3, 1]);
        end

        i = i + 1;
        gt1 = squeeze(gts(i, :, :, :));

        [q_avg_gt, sam_gt, ergas_gt, scc_gt, q_gt] = indexes_evaluation(sr1, gt1, ratio, L, Q_block_size, flag_cut_bounds, dim_cut, thvalues);
        [psnr_gt, ssim_gt] = quality_assess(sr1 / const, gt1 / const);

        sam(i) = sam_gt;
        ergas(i) = ergas_gt;
        scc(i) = scc_gt;
        qn(i) = q_gt;
        qave(i) = q_avg_gt;
        psnr(i) = psnr_gt;
        ssim(i) = ssim_gt;

        if printout
            fprintf("sample %d - sam: %f, ergas: %f, scc: %f, qn: %f, q_ave: %f, psnr: %f, ssim: %f \n", i, sam_gt, ergas_gt, scc_gt, q_gt, q_avg_gt, psnr_gt, ssim_gt)
        end

    end

    res.sam = [mean(sam), std(sam)];
    res.ergas = [mean(ergas), std(ergas)];
    res.scc = [mean(scc), std(scc)];
    res.qn = [mean(qn), std(qn)];
    res.qave = [mean(qave), std(qave)];
    res.psnr = [mean(psnr), std(psnr)];
    res.ssim = [mean(ssim), std(ssim)];

end
