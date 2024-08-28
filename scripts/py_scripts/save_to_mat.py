import h5py
import numpy as np
import os
import os.path as osp
import glob
import scipy.io as io
from tqdm import tqdm
import matplotlib.pyplot as plt


# file = r'/Data2/ZiHanCao/exps/panformer/visualized_img/data_dcformermwsa_new_cave_x8_ref.mat'


def process_mat_to_single_mat(file, show=True, force_all=True):
    name = file.split('/')[-1].strip('.mat')
    path = f'/Data2/ZiHanCao/exps/panformer/visualized_img/{name}'
    print(path)
    save_prefix = 'output_mulExm_'
    if not osp.exists(path) or force_all:
        os.makedirs(path, exist_ok=True)
        print(f'make dir {name}')
    else:
        print('already process this dir')
        return
        
    mat_file = io.loadmat(file)
    print(f'has keys: {mat_file.keys()}')
    
    if 'wv3' in path: 
        rgb_index = [4,2,0]
        const = 2047
    elif 'gf5' in path:
        rgb_index = [49, 39, 19]
        const = 1
    elif 'gf' in path or 'qb' in path: 
        rgb_index = [2,1,0]
        const = 1023
    elif 'cave' in path or 'harvard'in path:
        rgb_index = [29,19,9]
        const = 1
    elif 'houston' in path:
        rgb_index = [39, 29, 19]
        const = 1
    else:
        rgb_index = [29, 19, 9]
        const = 1

    sr = mat_file.get('sr')
    if show:
        ncols = int(np.ceil(sr.shape[0]/4))
        fig, axes = plt.subplots(4, ncols, figsize=(ncols*4, 4*4))
        axes = axes.flatten()
    
    if sr is None:
        print('has no key sr')
    else:
        bar = tqdm(range(sr.shape[0]))
        for i in bar:
            save_path = osp.join(path, save_prefix+f'{i}.mat')
            sr_i = np.transpose(sr[i, ...], [1,2,0])
            save_d = {'sr': sr_i}
            io.savemat(save_path, save_d)
            bar.set_description(f'save {i}.mat')
            
            if show:
                axes[i].imshow(sr_i[..., rgb_index] / const)
                axes[i].set_axis_off()
    if show:
        plt.tight_layout()
        fig.savefig(path + '/sr.png')
    print('---------------'*10)
            
# files = glob.glob('/Data2/ZiHanCao/exps/panformer/visualized_img/*.mat')

# print('found files: ', files)


# for file in files:
#     process_mat_to_single_mat(file, force_all=True)
    

        