import glob
import shutil
import re
import os

# cwd
os.chdir('/Data2/ZiHanCao/exps/panformer/weight')
print(os.getcwd())

inf_text = open('inference.py', 'r').readlines()
inf_fuse_text = open('inference_on_fuse.py', 'r').readlines()
inf_text = ''.join(inf_text)
inf_fuse_text = ''.join(inf_fuse_text)
re_pattern = r'[.|\/|a-z|\d_]+\.pth'

used_weight_paths = re.findall(re_pattern, inf_text)
used_weight_paths2 = re.findall(re_pattern, inf_fuse_text)

used_weight_paths = used_weight_paths + used_weight_paths2

os.makedirs('weight/used_weights', exist_ok=True)
print(f'moving the used weights...')

print('we found the used weights at:')
for p in used_weight_paths:
    if not os.path.exists(p):
        print('not found')
        print('-'*20)
        continue
    
    if os.path.dirname(p).endswith('weight'):
        print('found weight')
        print(p)
        shutil.move(p, 'weight/used_weights')
        
    else:
        print('found dir with weights')
        print(p)
        shutil.move(os.path.dirname(p), 'weight/used_weights')
        
    print('-'*20)
    
    

os.makedirs('weight/unused_weights', exist_ok=True)

all_weights = glob.glob('weight/*')
print(f'moving unused weights...')
for p in all_weights:
    if 'used_weights' not in p:
        shutil.move(p, 'weight/unused_weights')
    
    