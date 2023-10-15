import numpy as np
from PIL import Image 
import os 
from tqdm import tqdm 
import argparse

parser = argparse.ArgumentParser()  
parser.add_argument("--save_path", type=str, default='exp_3',
                        help='what exp number to save collague and load the corrected output from') 
args = parser.parse_args()

img_ids = []
list_path = '/home/sidd_s/mmsegmentation/work/dannetcode_lists/acdc_rgb.txt'
root_path = '/home/sidd_s/scratch/dataset'

with open(list_path) as f: 
    for item in f.readlines(): 
        fields = item.strip().split('\t')[0]
        if ' ' in fields:
            fields = fields.split(' ')[0]
        img_ids.append(fields)  

for name in tqdm(img_ids):
    nm = name.split('/')[-1]  

    ## gt  
    replace = (("_rgb_anon", "_gt_labelColor"), ("acdc_trainval", "acdc_gt"), ("rgb_anon", "gt")) 
    gt_name = name
    for r in replace: 
        gt_name = gt_name.replace(*r) 
    gpath = os.path.join(root_path, gt_name) 
    gt = np.array(Image.open(gpath))

    ## dannet output 
    dpath = os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/dannet_pred/val', nm) 
    dpred = np.array(Image.open(dpath))    
    
    ### original perturbed input 
    ppath = os.path.join('/home/sidd_s/scratch/dataset/acdc_trainval/rgb_anon/night/synthetic/val/1000n_20p_dannet_pred', nm.replace('_rgb_anon.png', '_gt_labelColor.png'))
    per = np.array(Image.open(ppath))
    
    ##### extrasss
    ## perturbed input 
    ppath1 = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/org', nm) 
    per1 = np.array(Image.open(ppath1))
    
    ## perturbed input 
    ppath2 = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/iter1', nm) 
    per2 = np.array(Image.open(ppath2))
    
    ## perturbed input 
    ppath3 = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/iter2', nm) 
    per3 = np.array(Image.open(ppath3))
    
    ## perturbed input 
    ppath4 = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/iter3', nm) 
    per4 = np.array(Image.open(ppath4))
    ##### extrasss

    ## perturbed input 
    ppath5 = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/iter4', nm) 
    per5 = np.array(Image.open(ppath5))

    ### perturbed input    
    ppath6 = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/iter5',nm) 
    per6 = np.array(Image.open(ppath6)) 

    ## collague way of saving images 
    
    imgh1 = np.hstack((gt,per1, per4)) 
    imgh2 = np.hstack((per, per2, per5))
    imgh3 = np.hstack((dpred, per3, per6)) 
    imgg = Image.fromarray(np.vstack((imgh1, imgh2, imgh3))) 
    path_col = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/collage/combine',nm) 

    # imgh1 = np.hstack((gt, per))
    # imgh2 = np.hstack((dpred, per1))
    # imgg = Image.fromarray(np.vstack((imgh1, imgh2))) 
    # if not os.path.exists(os.path.join('/home/sidd_s/mmsegmentation/collague/week26', args.save_path)):
    #     os.makedirs(os.path.join('/home/sidd_s/mmsegmentation/collague/week26', args.save_path))  
    # path_col = os.path.join('/home/sidd_s/scratch/mmseg_results/week28/collage/org',nm) 
    # print(path_col)
    imgg.save(path_col)