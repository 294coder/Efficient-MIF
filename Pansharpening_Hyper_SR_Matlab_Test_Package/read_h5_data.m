function [ms, lms, pan, sr, gt] = read_h5_data(path)
    ms = h5read(path, '/ms');
    lms = h5read(path, '/lms');
    pan = h5read(path, '/pan');
    sr = h5read(path, '/sr');
    gt = h5read(path, '/gt');

    ms = permute(ms, [4, 3, 1, 2]);
    lms = permute(lms, [4, 3, 1, 2]);
    pan = permute(pan, [4, 3, 1, 2]);
    sr = permute(sr, [4, 3, 1, 2]);
    gt = permute(gt, [4, 3, 1, 2]);
end