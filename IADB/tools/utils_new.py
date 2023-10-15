import os 
from tqdm import tqdm

path='/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/1000n_20p_dannet_pred'  
for file in tqdm(os.listdir(path)):   
    old_name = os.path.join(path, file)
    os.rename(old_name, old_name.replace('_rgb_anon.png', '_gt_labelColor.png')) 