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
from test_softmax_pred import main 
from mmcv.cnn import ConvModule
import mmcv 
from tqdm import tqdm

## condition => the "softmax-logits" prediction of cityscapes dataset from segformer model 
## we will be loading trained model, so the configuration will be that of validation of mmseg model 
segformer_model_path = '/home/sidd_s/scratch/saved_models/mmseg/segformer_b2_cityscapes_1024x1024/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'
config_file_path = '/home/sidd_s/scratch/saved_models/mmseg/segformer_b2_cityscapes_1024x1024/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py' 
results_softmax_predictions_train, results_softmax_predictions_val = main(config_path= config_file_path, checkpoint_path= segformer_model_path) # lets check!  ## caching in train and val data softmax predictions; so that segformerb2 not have to predict every other data instant but rather can be easily indexed for softmax prediction generation.
print('results consisting of softmax predictions loaded successfully!')

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
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=20, in_channels=20, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

def label_img_to_color(img: torch.tensor): 
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


@torch.no_grad() 
def sample_conditional_seg_iadb(model, datav, nb_step, device, num_classes, conditional_transform): # arguments as: de-blending model, and neighbouring steps for deblending operation
    model = model.eval() 
    softmax_feats = torch.tensor(results_softmax_predictions_val[datav[2][0]]).to(device).unsqueeze(dim=0) ## since batch size is 1
    extended_softmax_feats = torch.rand((softmax_feats.shape[0], num_classes, *tuple(softmax_feats.shape[2:])), device=device)  ## for including background
    extended_softmax_feats[:, :softmax_feats.shape[1], :, :] = softmax_feats # B,C,H,W ## C = 20 (including background) 
    ## x0 as the stationary distribution 
    x0 = torch.randn_like(extended_softmax_feats)
    ## conditional input 
    conditional_feats = torch.cat([extended_softmax_feats, x0], dim=1)
    conditional_feats = conditional_transform(conditional_feats)
    
    ## now deblending starts: 
    x_alpha = conditional_feats
    for t in tqdm(range(nb_step)):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(x_alpha, torch.tensor(alpha_start, device=x_alpha.device))['sample']
        x_alpha = x_alpha + (alpha_end-alpha_start)*d

    return x_alpha
    



## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self, img_dir = '/home/sidd_s/scratch/dataset/cityscapes/leftImg8bit/' , img_transform = None, gt_dir = "/home/sidd_s/scratch/dataset/cityscapes/gtFine/", suffix = '_gtFine_labelTrainIds.png', lb_transform = None, num_classes = 20, mode = 'train'):
        self.img_transform = img_transform
        self.img_data_list = []
        self.gt_dir = gt_dir + mode  
        self.img_dir = img_dir + mode
        self.lb_transform = lb_transform 
        self.data_list = []
        self.num_classes = num_classes # 19 + background class 
        
        
        for root, dirs, files in os.walk(self.gt_dir, topdown=False):
            for name in tqdm(sorted(files)):
                path = os.path.join(root, name)
                if path.find(suffix)!=-1:
                    self.data_list.append(path)

        for root, dirs, files in os.walk(self.img_dir, topdown=False):
            for name in tqdm(sorted(files)):
                img_path = os.path.join(root, name)
                self.img_data_list.append(img_path)


        if mode == 'train':
            assert len(self.data_list) == 2975 == len(self.img_data_list)
        elif mode == 'val':
            assert len(self.data_list) == 500 == len(self.img_data_list)
        else:
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.data_list) 
    
    def __getitem__(self, index):

        img_path = self.img_data_list[index] 
        img = Image.open(img_path)
        if self.img_transform:
            img = self.img_transform(img)

        label_path = self.data_list[index]  
        label = torch.tensor(np.array(Image.open(label_path)))
        label[label==255] = 19
        if self.lb_transform: 
            label = self.lb_transform(label.unsqueeze(dim=0)) # resizing the tensor, for working in low dimension
        
        return img, label, img_path
       

def main(): 
    print('in the main function')
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    gt_dir = '/home/sidd_s/scratch/dataset/cityscapes/gtFine/' 
    img_dir = '/home/sidd_s/scratch/dataset/cityscapes/leftImg8bit/'
    suffix = "_gtFine_labelTrainIds.png"
    num_classes = 20  
    lb_transform = transforms.Compose([ 
        transforms.Resize((256,512), interpolation=InterpolationMode.NEAREST), # (H/4, W/4) 
    ])
    img_transform = transforms.Compose([ 
        transforms.Resize((256,512)), # (H/4, W/4) 
        transforms.ToTensor(), 
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), # imagenet mean and std using
    ])
    conditional_transform = ConvModule(
            num_classes * 2,
            num_classes,
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ).to(device)
    ## train dataloader 
    dataset_train = custom_cityscapes_labels(img_dir, img_transform, gt_dir,suffix, lb_transform,num_classes, mode = 'train')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=0, drop_last=True)  
    ## val dataloader
    dataset_val = custom_cityscapes_labels(img_dir, img_transform, gt_dir,suffix, lb_transform,num_classes, mode = 'val')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=0, drop_last=True)  
    print('dataset loaded successfully!')

    model = get_model() 
    model = model.to(device)
    print('Model loaded into cuda')

    optimizer = Adam(model.parameters(), lr=1e-4)
    nb_iter = 0
    best_loss = torch.finfo(torch.float32).max # init the best loss 
    print('Start training')
    for _ in tqdm(range(100)):
        for i, data in enumerate(dataloader_train):
            ## >>x1 being the target distribution<<
            labels_one_hot = F.one_hot(data[1].squeeze().long(), num_classes)
            labels_one_hot = labels_one_hot.permute(0,3,1,2) # B, C, H, W ## in order to present any correct gt label, I have to proceed with one-hot (rather than logits type)
            x1 = (labels_one_hot.to(device)*2)-1  # acc to original IADB ## => [-1,1]
            
            ## x0 being the stationary distribution! 
            x0 = torch.randn_like(x1.float()) # standard normal distribution  # acc to original IADB 
            ## conditioning is done in terms of softmax-logits of a model(which to be improved)
            c  = [torch.tensor(results_softmax_predictions_train[path]) for path in data[2]] # conditioning softmax prediciton
            c = torch.stack(c) # B,C,H,W ## here C = 19
            extended_c = torch.rand(x1.shape, device=device)  ## for including background
            extended_c[:, :c.shape[1], :, :] = c # B,C,H,W ## C = 20 (including background) 
            extended_c = (extended_c * 2) - 1 ## similar to x1 => [-1,1]

            ## conditional input ## conditioning is done similar to DDP & DDPS model!
            conditional_feats = torch.cat([extended_c, x0], dim=1)
            conditional_feats = conditional_transform(conditional_feats) ## not doing anything cause, it will learn from itself to what value it should adjust to, while the learning of the whole objective is being carried out.
             
            bs = x0.shape[0] # batch size 

            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * conditional_feats
            
            d = model(x_alpha, alpha)['sample'] ## model involved for denoising/(de-blending here), the blended value.
            loss = torch.sum((d - (x1-conditional_feats))**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nb_iter += 1

            if nb_iter % 1 == 0:  ## testing remainder factor by making it to 0 instead of 200
                print('In Sampling')
                dataset = dataloader_val.dataset
                prog_bar = mmcv.ProgressBar(len(dataset))
                results = [] 
                save_imgs_dir = '/home/sidd_s/scratch/saved_models/iadb_cond_seg/result_val_images'
                for __, datav in enumerate(dataloader_val):
                    with torch.no_grad(): 
                        ## rather than the below commented code, I believe I think I should take softmax
                        # sample = (sample_conditional_seg_iadb(model, datav, nb_step=128) * 0.5) + 0.5 ## converting back to 0 to 1 | from [-1,1] 
                        x1_sample = sample_conditional_seg_iadb(model, datav, nb_step=128, device=device, num_classes=20, conditional_transform=conditional_transform)
                        x1_sample = F.softmax(x1_sample, dim=1)
                        argmax_x1_sample = torch.argmax(x1_sample, dim=1) 
                        results.append(argmax_x1_sample) 
                        save_path = os.path.join(save_imgs_dir, datav[2][0].split('/')[-1].replace('_leftImg8bit.png', '_predFine_color.png'))
                        x1_sample_color = Image.fromarray(label_img_to_color(argmax_x1_sample.cpu()))
                        x1_sample_color.save(save_path)
                        prog_bar.update()
                        
                        if loss.item() < best_loss:
                            best_loss = loss
                            torch.save(model.state_dict(), f'/home/sidd_s/scratch/saved_models/iadb_cond_seg/best_model_parameters.pt')
                            print('Model updated! : current best model saved') 
                        

if __name__ == '__main__':
    main()