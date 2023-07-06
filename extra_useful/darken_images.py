import os
from PIL import Image, ImageEnhance
import numpy as np 
from tqdm import tqdm

# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/dataset/cityscapes/leftImg8bit/val"  
    factor = 0.5
    
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in tqdm(files): 
            path = os.path.join(root, name) 
            new_folder = root.replace('/leftImg8bit/', '/leftImg8bit_darken_brightness' + str(factor) +  '/') 
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            img = Image.open(path)
            
            dst = f"{new_folder}/{name}" 
            brightness = ImageEnhance.Brightness(img)
            brightness.enhance(factor).save(dst) 
        
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()