import os
from PIL import Image, ImageEnhance
import numpy as np 
from tqdm import tqdm

def label_img_to_color(img): 
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [0,  0, 0] 
        } 
    img = np.array(img.squeeze()) 
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(label_to_color[label])  
    return img_color


# Function to rename multiple files
def main():
   
    folder = "/home/sidd_s/scratch/dataset/cityscapes/pred/segformerb2/res_128x256"  
    factor = 0.2
    
    for root, dirs, files in os.walk(folder, topdown=False):
        for name in tqdm(files): 
            path = os.path.join(root, name) 
            new_folder = root.replace('/leftImg8bit/', '/leftImg8bit_darken_contrast_' + str(factor) +  '/') 
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)
            img = Image.open(path)
            dst = f"{new_folder}/{name}" 
            # brightness = ImageEnhance.Brightness(img)
            # brightness.enhance(factor).save(dst) 
            contrast = ImageEnhance.Contrast(img)
            contrast.enhance(factor).save(dst)
            # break 
        
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()