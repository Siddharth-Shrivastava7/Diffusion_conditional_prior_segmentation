import os
from PIL import Image 
import numpy as np 
from tqdm import tqdm

# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/dataset/dark_zurich_val_morepred/gt/val/night/GOPR0356"
    for filename in tqdm(os.listdir(folder)):
        if filename.find('_gt_labelTrainIds.png')!=-1: 
            src =f"{folder}/{filename}" 
            new_folder = folder.replace('val/night/GOPR0356', 'resized_val_256x512')
            # new_filename = filename.replace('', '')
            dst = f"{new_folder}/{filename}"
            # print(dst)
            if not os.path.exists(new_folder):
                # print('>>>>>>>>>') 
                os.makedirs(new_folder)
            ## for pallete single channel label map images to resize 
            # im = Image.open(src)
            # im = im.resize((512, 256))
            # im.save(dst, "PNG")
            
            ## for single channel label map images to resize 
            im = Image.open(src)
            im = im.resize((512, 256), Image.NEAREST)
            im.save(dst, "PNG")
        else:
            continue 

# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()