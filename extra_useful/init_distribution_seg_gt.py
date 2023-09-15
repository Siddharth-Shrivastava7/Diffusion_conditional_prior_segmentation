import os
from PIL import Image, ImageEnhance
import numpy as np 
from tqdm import tqdm

# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/results/segformer/cityscapes/val"  
    gt_dataset_num_of_labels = dict.fromkeys(range(19),0) ## 19 classes, if finding the distribution of model prediction rather than gt
    num_gts = 0
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in tqdm(files): 
            path = os.path.join(root, name) 
            if path.find('_leftImg8bit.png')!=-1:
                num_gts += 1
                gt = np.array(Image.open(path))
                # gt[gt==255] = 19 ## for background class label 
                unique_labels, unique_labels_counts = np.unique(gt, return_counts = True)
                assert unique_labels_counts.sum() == 1024*2048
                for ind in range(unique_labels.shape[0]):
                    gt_dataset_num_of_labels[unique_labels[ind]] += unique_labels_counts[ind]
    
    assert num_gts == 500
    
    ''' 
        now finding probability distribution which will serve as init distribution over the gts 
    '''    
    
    total_num_pixels = gt.reshape(1, -1).shape[1] * num_gts 
    gt_dataset_labels_init_distribution = {k: v / total_num_pixels for k, v in gt_dataset_num_of_labels.items()}
    # save in numpy format
    np.save('segformerb2_cityscapes_pred_val_distribution.npy', gt_dataset_labels_init_distribution)
    # loading dictionary 
    # cityscapes_gt_labels_init_distribution = np.load('cityscapes_gt_labels_init_distribution.npy',allow_pickle='TRUE').item()
    print(gt_dataset_labels_init_distribution) 
    print('Finished it')    
                
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()