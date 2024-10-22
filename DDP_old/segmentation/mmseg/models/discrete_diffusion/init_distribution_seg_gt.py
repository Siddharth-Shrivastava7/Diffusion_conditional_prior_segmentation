import os
from PIL import Image, ImageEnhance
import numpy as np 
from tqdm import tqdm

# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/dataset/cityscapes/gtFine/train"
    # gt_dataset_num_of_labels = dict.fromkeys(range(20),0) # with background 
    gt_dataset_num_of_labels = dict.fromkeys(range(19),0) 
    num_gts = 0
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in tqdm(files): 
            path = os.path.join(root, name) 
            if path.find('_gtFine_labelTrainIds.png')!=-1:
                num_gts += 1
                gt = np.array(Image.open(path))
                # gt[gt==255] = 19 ## for background class label # with background
                '''
                    not taking into account background  
                '''
                gt_flat = gt.flatten()     
                mask = gt_flat!=255 
                unique_labels, unique_labels_counts = np.unique(gt_flat[mask], return_counts = True) 
                
                # unique_labels, unique_labels_counts = np.unique(gt, return_counts = True) # with background
                # assert unique_labels_counts.sum() == 1024*2048 # with background 
                for ind in range(unique_labels.shape[0]):
                    gt_dataset_num_of_labels[unique_labels[ind]] += unique_labels_counts[ind]
    
    assert num_gts == 2975
    
    ''' 
        now finding probability distribution which will serve as init distribution over the gts 
    '''    
    
    # total_num_pixels = gt.reshape(1, -1).shape[1] * num_gts  # with background 
    # gt_dataset_labels_init_distribution = {k: v / total_num_pixels for k, v in gt_dataset_num_of_labels.items()} # with background 
    gt_dataset_labels_init_distribution = {k: v / sum(gt_dataset_num_of_labels.values(), 0.0) for k, v in gt_dataset_num_of_labels.items()}
    assert np.rint(sum(gt_dataset_labels_init_distribution.values(), 0.0)) == 1  ## for a valid probability distribution 
    # save in numpy format
    # np.save('cityscapes_gt_labels_init_distribution.npy', gt_dataset_labels_init_distribution) # with background
    np.save('cityscapes_gt_labels_init_distribution_without_background.npy', gt_dataset_labels_init_distribution)
    # loading dictionary 
    # cityscapes_gt_labels_init_distribution = np.load('cityscapes_gt_labels_init_distribution.npy',allow_pickle='TRUE').item()
    print(gt_dataset_labels_init_distribution) 
    print('Finished it')    
                
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()