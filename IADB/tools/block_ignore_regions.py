import numpy as np
from PIL import Image
import os 
from tqdm import tqdm

gt_path = '/home/sidd_s/scratch/dataset/acdc_gt/gt/night/val'

gt_label_list = []
for dir in tqdm(os.listdir(gt_path)):
    for file in os.listdir(os.path.join(gt_path, dir)):
        if '_gt_labelTrainIds' in file: 
            gt_label_list.append(os.path.join(gt_path,dir, file)) 
gt_label_list = sorted(gt_label_list)

pred_path = '/home/sidd_s/scratch/mmseg_results/week28/iter5' 
pred_list = sorted([os.path.join(pred_path,p) for p in os.listdir(pred_path)])

for ind in tqdm(range(len(pred_list))):  
    gt = np.array(Image.open(gt_label_list[ind]))
    pred = np.array(Image.open(pred_list[ind]))
    mask = (gt == 255)
    pred[mask] = (0,0,0) 
    pred_img = Image.fromarray(pred) 
    pred_img.save(pred_list[ind])