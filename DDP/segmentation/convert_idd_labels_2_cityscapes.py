import os 
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

def main():
    MAP_dict = {0: 0, 1: 255, 2: 1, 3: 9, 4: 11, 5: 12, 6: 17, 7: 18, 8: 255, 9: 13, 10: 14, 11: 15, 12: 255, 13: 255, 14: 3,
                    15: 4, 16: 255, 17: 255, 18: 7, 19: 6, 20: 5, 21: 255, 22: 2, 23: 255, 24: 8, 25: 10, 255: 255}
    k = np.array(list(MAP_dict.keys()))
    v = np.array(list(MAP_dict.values()))
    # print(sorted(k), sorted(v))
    folder = "/home/sidd_s/scratch/dataset/idd20k_final/valid/labels" # idd labels folder
    for count, filename in tqdm(enumerate(os.listdir(folder))):
        src =f"{folder}/{filename}" 
        label = np.array(Image.open(src))
        mapping_ar = np.zeros(k.max()+1, dtype=v.dtype)
        mapping_ar[k] = v  # dict bananyi, k mei dalo, v mei chala jaega
        label = mapping_ar[label]  # map label to new values of IDD label
        print('>>>>>>>>>>', label.shape, np.unique(label))
        # plt.imsave(src.replace('labels','labels_trainids'), np.uint8(label), cmap='gray')
        # cv2.imwrite(src.replace('labels','labels_trainids'), label)
        label = Image.fromarray(np.uint8(label))
        label.save(src.replace('labels','labels_trainids'))
        # break
        
        
        
        
        
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()