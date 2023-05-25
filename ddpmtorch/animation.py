import numpy as np
import imageio 
import os 
import torch 

from PIL import Image

src_path = '/home/sidd_s/scratch/results/ddpm/images/train/cifar10'
files_names = os.listdir(src_path)

images = []
for fn in files_names:
    fpath = os.path.join(src_path, fn)
    image = torch.tensor(np.array(Image.open(fpath)))
    images.append(image)

imageio.mimsave('ddpm_chain.gif', images) 