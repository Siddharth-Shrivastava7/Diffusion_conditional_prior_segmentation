from PIL import Image
import os
import pickle
from glob import glob 
import numpy as np
from tqdm import tqdm 

## to convert whole images inside bird folder
## input folder = bird\\images\\all_images_in_jpg_format

# PICKLE_FILE = "/home/sidd_s/scratch/results/oneformer/results_filenames.pickle"
SOURCE_DIRECTORY = "/home/sidd_s/scratch/results/deeplabv3+r50/cityscapes/"
PICKLE_IMAGES = "/home/sidd_s/scratch/results/deeplabv3+r50/cityscapes/results_images.pickle"

path_list = sorted(glob(os.path.join(SOURCE_DIRECTORY, "*.png")))

# pickle images into big pickle file
list_images = []

with open(PICKLE_IMAGES,"wb") as f:
    for file_name in tqdm(path_list):
        list_images.append(np.array(Image.open(file_name)))
    pickle.dump(list_images,f)
    
print('done')
        
# # get short names from the path list 

# file_list = list(
#     map(
#         lambda x: os.path.basename(x), path_list)
# )

# # pickle short name list

# pickle.dump(file_list, open(PICKLE_FILE, 'wb'))

# # test that we can reread the list

# recovered_list = pickle.load(open(PICKLE_FILE,"rb"))

# if file_list == recovered_list:
#     print("Lists Match!")
# else:
#     print("Lists Don't Match!!!")


# read a couple images out of the image file:

# display_count = 5


# with open(PICKLE_IMAGES,"rb") as f:
#     while True:
#         try:
#             pickle.load(f).show()
#             display_count -= 1
#             if display_count <= 0:
#                 break
#         except EOFerror as e:
#             break