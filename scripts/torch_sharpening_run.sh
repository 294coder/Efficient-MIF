## MIMO_SST arch
python main.py --proj_name MIMO_SST --arch MIMO_SST \
-b 32 --device 'cuda:0' --dataset 'wv3' \
--warm_up_epochs 10 --num_worker 6 -e 2000 --aug_probs 0. 0. \
--loss l1ssim --val_n_epoch 20 --comment 'MIMO_SST reduced arch on wv3 dataset' \
--log_metrics \
--logger_on \
# --pretrain --pretrain_id 'd02s63w5'
