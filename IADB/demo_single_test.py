'''
conditional image segementation map generation, using alpha bending of gaussian and cityscapes gt maps, with conditioned on segformer softmax prediction>>later will replace unet with transformers of ddp>>conditioning is achieved using concatenation here
'''

import torch
import torchvision
from torchvision import transforms
from diffusers import UNet2DModel
from torch.optim import Adam 
from tqdm import tqdm 
from torch.utils.data import Dataset
import os
from tqdm import tqdm
from PIL import Image 
import numpy as np
from  torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F

## for now, not iterating over the whole dataset but rather demo testing on one single image! 
test_gt_label_path = '/home/sidd_s/scratch/dataset/cityscapes/gtFine/train/hamburg/hamburg_000000_000042_gtFine_labelTrainIds.png' # this will be x1 


def get_model():
    block_out_channels=(128, 128, 256, 256, 512, 512)
    down_block_types=( 
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D", 
        "DownBlock2D", 
        "DownBlock2D", 
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    )
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D", 
        "UpBlock2D"  
    )
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=3, in_channels=3, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

@torch.no_grad()
def sample_iadb(model, x0, nb_step):
    x_alpha = x0
    for t in range(nb_step):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha

## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self, gt_dir = "/home/sidd_s/scratch/dataset/cityscapes/gtFine/", suffix = '_gtFine_labelTrainIds.png', transform = None, mode = 'train', num_classes = 20):
        self.gt_dir = gt_dir 
        self.transform = transform 
        self.data_list = []
        self.mode = mode 
        self.gt_dir_mode = self.gt_dir + self.mode  
        self.num_classes = num_classes # 19 + background class 
        
        for root, dirs, files in os.walk(self.gt_dir_mode, topdown=False):
            for name in tqdm(files):
                path = os.path.join(root, name)
                if path.find(suffix)!=-1:
                    self.data_list.append(path)

        if mode == 'train':
            assert len(self.data_list) == 2975
        elif mode == 'val':
            assert len(self.data_list) == 500
        else:
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, index):

        label_path = self.data_list[index]  
        label = torch.tensor(np.array(Image.open(label_path)))
        if self.transform: 
            label = self.transform(label) # resizing the tensor, for working in low dimension
        label_one_hot = F.one_hot(label, self.num_classes)

        return label_one_hot

## condition => the softmax prediction of cityscapes dataset from segformer model 



def main(): 
    print('in the main function')
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    gt_dir = '/home/sidd_s/scratch/dataset/cityscapes/gtFine/' 
    mode  = 'val'
    num_classes = 20 
    transform = transforms.Compose([ 
        transforms.Resize((256,512), interpolation=InterpolationMode.NEAREST), # (H/4, W/4) 
    ])
    dataset = custom_cityscapes_labels(gt_dir, transform, mode, num_classes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True) 
    print('dataset loaded successfully!')

    model = get_model() 
    model = model.to(device)
    print('Model loaded into cuda')

    optimizer = Adam(model.parameters(), lr=1e-4)
    nb_iter = 0
    print('Start training')
    for _ in tqdm(range(100)):
        for i, data in enumerate(dataloader):
            x1 = (data[0].to(device)*2)-1
            x0 = torch.randn_like(x1)
            bs = x0.shape[0]

            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0
            
            d = model(x_alpha, alpha)['sample']
            loss = torch.sum((d - (x1-x0))**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_iter += 1

            if nb_iter % 200 == 0: 
                print('In Sampling')
                with torch.no_grad():
                    print(f'Save export {nb_iter}')
                    sample = (sample_iadb(model, x0, nb_step=128) * 0.5) + 0.5
                    torchvision.utils.save_image(sample, f'/home/sidd_s/scratch/saved_models/iadb/sample_imgs/export_{str(nb_iter).zfill(8)}.png')
                    torch.save(model.state_dict(), f'/home/sidd_s/scratch/saved_models/iadb/celeba.ckpt')


if __name__ == '__main__':
    main()