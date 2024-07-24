function res = analysis_unref_batched_images(path, ratio, sensor)
    if strcmp(sensor, 'QB') || strcmp(sensor, 'GF2')
        sensor = 'IKONOS';
    end
    disp(sensor)
    data=load(path);
    sr = data.sr;
    ms = data.ms;
    lms = data.lms;
    pan = data.pan;

    addpath('./Quality_Indices/')
    addpath('./Tools')
    sz = size(sr);
    bs = sz(1);
    d_lambdas = [];
    qnr_indices = [];
    d_ses = [];
    blockSize = 32;
    res = {};
    h = size(pan, 3);
    w = size(pan, 4);
    for i = (1: bs) 
        sr1 = permute(squeeze(sr(i, :, :, :)), [2, 3, 1]);
        ms1 = permute(squeeze(ms(i, :, :, :)), [2, 3, 1]);
        lms1 = permute(squeeze(lms(i, :, :, :)), [2, 3, 1]);
        pan1 = permute(reshape(squeeze(pan(i, :, :, :)), 1, h, w), [2, 3, 1]);
        [d_lambdas(i), d_ses(i), qnr_indices(i)] = indexes_evaluation_FS(sr1, ms1, pan1, 11, 0, lms1, sensor, ratio, 0);
        fprintf('sample %d - d_lambda: %f, qnr_index: %f, d_s: %f \n', i, d_lambdas(i), qnr_indices(i), d_ses(i))
    end
    res.d_lambda = [mean(d_lambdas), std(d_lambdas)];
    res.qnr_index = [mean(qnr_indices), std(qnr_indices)];
    res.d_s = [mean(d_ses), std(d_ses)];
end