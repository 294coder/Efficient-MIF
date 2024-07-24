function res = analysis_unref_unbatched_images(dir_path, ratio, sensor)
    full_data_path = "/Data2/ZiHanCao/datasets/pansharpening/pansharpening_test/test_wv3_OrigScale_multiExm1.h5";
    sensor = lower(sensor);

    % if sensor == "wv3" || sensor == "wv2" || sensor == "qb"
    %     const = 2047;
    % elseif sensor == "gf2" || sensor == "gf"
    %     const = 1023;
    % else
    %     error(strcat(sensor, ' is not supported!'))
    
    ms_s = h5read(full_data_path, '/ms');
    ms_s = permute(ms_s, [4,2,1,3]);
    lms_s = h5read(full_data_path, '/lms');
    lms_s = permute(lms_s, [4,2,1,3]);
    pan_s = h5read(full_data_path, '/pan');
    pan_s = permute(pan_s, [4,2,1,3]);
    bs = size(lms_s, 1);

    addpath('./Tools')
    addpath('./Quality_Indices/')

    d_lambdas = [];
    qnr_indices = [];
    d_ses = [];

    res = {};
    h = size(pan_s, 3);
    w = size(pan_s, 4);

    for i = (0:bs-1)
        p = strcat(dir_path, "/", "output_mulExm_", string(i), ".mat");
        sr2 = load(p);
        sr = sr2.sr;
        
        i = i+1;
        ms = squeeze(ms_s(i, :, :, :));
        lms = squeeze(lms_s(i, :, :, :));
        pan = squeeze(pan_s(i, :, :, :));

        [d_lambdas(i), d_ses(i), qnr_indices(i)] = indexes_evaluation_FS(sr, ms, pan, 11, 0, lms, sensor, ratio, 0);

        fprintf('sample %d - d_lambda: %f, qnr_index: %f, d_s: %f \n', i, d_lambdas(i), qnr_indices(i), d_ses(i))

        res.d_lambda = [mean(d_lambdas), std(d_lambdas)];
        res.qnr_index = [mean(qnr_indices), std(qnr_indices)];
        res.d_s = [mean(d_ses), std(d_ses)];

    end

    res.d_lambda = [mean(d_lambdas), std(d_lambdas)];
    res.qnr_index = [mean(qnr_indices), std(qnr_indices)];
    res.d_s = [mean(d_ses), std(d_ses)];
end
