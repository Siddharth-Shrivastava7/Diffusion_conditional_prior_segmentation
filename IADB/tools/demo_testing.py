'''
conditional image segementation map generation, using alpha bending of gaussian and cityscapes gt maps, with conditioned on segformer softmax prediction>>later will replace unet with transformers of ddp>>conditioning is achieved using concatenation here
'''

import torch
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
import torch.nn as nn

torch.backends.cudnn.benchmark = True ## for better speed 

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
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=num_classes+1, in_channels=num_classes + 1, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

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


@torch.no_grad() 
def sample_conditional_seg_iadb(model, datav, conditional_transform, embedding_table, bit_scale, device, nb_step):
    model = model.eval() 
    conditional_transform = conditional_transform.eval() 
    
    ## predictions of segformerb2 as the conditions
    pred_label_emdb = embedding_table(datav[1].long().to(device, non_blocking=True)).squeeze(1).permute(0, 3, 1, 2) 
    pred_label_emdb = (torch.sigmoid(pred_label_emdb)*2 - 1)*bit_scale ## sort of logits
    
    ## x0 as the stationary distribution 
    x0 = torch.randn_like(pred_label_emdb) ## sort of logits

    ## conditional input 
    conditional_feats = torch.cat([pred_label_emdb, x0], dim=1)
    conditional_feats = conditional_transform(conditional_feats) ## sort of logits
    
    ## now deblending starts: 
    x_alpha = x0 # our stationary distribution is Gaussian distribution only! 
    for t in tqdm(range(nb_step)):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(conditional_feats, torch.as_tensor(alpha_start, device=x_alpha.device))['sample'] ## this is giving ~ (\bar_{x1} - \bar{x0})
        x_alpha = x_alpha + (alpha_end-alpha_start)*d 

        conditional_feats = torch.cat([pred_label_emdb, x_alpha], dim=1)
        conditional_feats = conditional_transform(conditional_feats)

    return x_alpha
    

## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self, pred_dir = '/home/sidd_s/scratch/dataset/cityscapes/pred/segformerb2/', gt_dir = "/home/sidd_s/scratch/dataset/cityscapes/gtFine/", suffix = '_gtFine_labelTrainIds.png', lb_transform = None, mode = 'train'):
        self.pred_list = []
        self.pred_dir = pred_dir + mode
        self.gt_dir = gt_dir + mode  
        self.lb_transform = lb_transform 
        self.label_list = []
        
        for root, dirs, files in os.walk(self.gt_dir, topdown=False):
            for name in tqdm(sorted(files)):
                path = os.path.join(root, name)
                if path.find(suffix)!=-1:
                    self.label_list.append(path) 
                    city_name = path.split('/')[-2] 
                    pred_path = path.replace('/gtFine/' + mode + '/' + city_name, '/pred/segformerb2/' + mode).replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')
                    self.pred_list.append(pred_path)

        if mode == 'train':
            assert len(self.label_list) == 2975 == len(self.pred_list)
        elif mode == 'val':
            assert len(self.label_list) == 500 == len(self.pred_list)
        else:
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.label_list) 
    
    def __getitem__(self, index):

        pred_path = self.pred_list[index] 
        pred_label = torch.from_numpy(np.array(Image.open(pred_path)))

        label_path = self.label_list[index]  
        label = torch.from_numpy(np.array(Image.open(label_path)))
        label[label==255] = 19 # for background label 

        if self.lb_transform: 
            label = self.lb_transform(label.unsqueeze(dim=0)) # resizing the tensor, for working in low dimension
            pred_label = self.lb_transform(pred_label.unsqueeze(dim=0)) ## Two cases emerge here: 1. resizing segformer output to 128x256, when its input image was 1024x2048 2. resizing the input to 128x256 and then use model prediction resizing at a reduced size, which was in the segformer model was upsampled in a biliner interpolation fashion on its logits. =>> for now, going with 2. way, since we might find improvement earlier here. 
        
        return label, pred_label, pred_path
       

def main(): 
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    gt_dir = '/home/sidd_s/scratch/dataset/cityscapes/gtFine/' 
    pred_dir = '/home/sidd_s/scratch/dataset/cityscapes/pred/segformerb2/res_128x256' ## opting for option 2, as written above, near below self.lb_transform usecase.
    suffix = "_gtFine_labelTrainIds.png"
    global num_classes
    num_classes = 19 ## only foreground classes 
    embed_dim = num_classes + 1 #a hyper-param ##used in order to arrive at a consistency with DDP and overcome the issue of random assignment for background
    embedding_table = nn.Embedding(num_classes + 1, embedding_dim=embed_dim).to(device)
    bit_scale = 0.01 #a hyper-param ##similar to DDP
    lb_transform = transforms.Compose([ 
        transforms.Resize((128,256), interpolation=InterpolationMode.NEAREST), # (H/4, W/4) ## similar to DDP, where they were training using (512, 1024) images and performed diffusion in (H/4, W/4) => (128, 256) and then used bilinear upsampling of the logits to obtain final prediction label.
    ])
    conditional_transform = ConvModule(
            embed_dim * 2,
            embed_dim,
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ).to(device) ## similar to DDP 
    ## train dataloader 
    dataset_train = custom_cityscapes_labels(pred_dir, gt_dir, suffix, lb_transform, mode = 'train')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=8, shuffle=True, num_workers=4, drop_last=True, pin_memory = True)  
    ## val dataloader
    dataset_val = custom_cityscapes_labels(pred_dir, gt_dir, suffix, lb_transform, mode = 'val')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4, pin_memory = True)  
    print('dataset loaded successfully!')

    model = get_model() 
    model = model.to(device)
    print('Model loaded into cuda')

    optimizer = Adam(model.parameters(), lr=1e-4) # hyper-param
    best_loss = torch.finfo(torch.float32).max # init the best loss 
    print('Start training')
    for epoch in tqdm(range(100)):
        for i, data in tqdm(enumerate(dataloader_train)):
            ## >>x1 being the target distribution<<
            ### similar to one done in DDP
            x1 = embedding_table(data[0].long().to(device, non_blocking=True)).squeeze(1).permute(0,3,1,2)
            x1 = (torch.sigmoid(x1)*2 - 1)*bit_scale ## sort of logits (encoding of discrete labels)

            ## x0 being the stationary distribution! 
            x0 = torch.randn_like(x1) # standard normal distribution  # acc to original IADB ## sort of logits
            
            bs = x0.shape[0] # batch size 
            ## alpha blending taking place between x0 (not conditional feats!) and x1 
            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0

            ## similar to DDP 
            pred_labels_emdb = embedding_table(data[1].long().to(device, non_blocking=True)).squeeze(1).permute(0, 3, 1, 2) 
            pred_labels_emdb = (torch.sigmoid(pred_labels_emdb)*2 - 1)*bit_scale ## sort of logits

            ## condition input 
            conditional_feats = torch.cat([pred_labels_emdb, x_alpha], dim=1)
            conditional_feats = conditional_transform(conditional_feats)
             
            bs = x0.shape[0] # batch size 

            d = model(conditional_feats, alpha)['sample'] ## model involved for denoising/(de-blending here), the blended value.
            loss = torch.sum((d - (x1-x0))**2) ## based on IADB paper

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    
        print('In Sampling at epoch:' + str(epoch+1))
        dataset = dataloader_val.dataset
        prog_bar = mmcv.ProgressBar(len(dataset))
        save_imgs_dir = '/home/sidd_s/scratch/saved_models/iadb_cond_seg/result_val_images'
        save_imgs_dir_ep = os.path.join(save_imgs_dir, str(epoch+1))
        if not os.path.exists(save_imgs_dir_ep):
            os.makedirs(save_imgs_dir_ep)
        for __, datav in enumerate(dataloader_val):
            with torch.no_grad(): 
                x1_sample_logits = sample_conditional_seg_iadb(model, datav, conditional_transform, embedding_table, bit_scale, device, nb_step=128) ## nb_step is a hyper-param > taken from IADB 
                x1_sample_softmax = F.softmax(x1_sample_logits, dim=1)
                argmax_x1_sample = torch.argmax(x1_sample_softmax, dim=1) 
                save_path = os.path.join(save_imgs_dir_ep, datav[2][0].split('/')[-1].replace('_leftImg8bit.png', '_predFine_color.png'))
                x1_sample_color = Image.fromarray(label_img_to_color(argmax_x1_sample.detach().cpu()))
                x1_sample_color.save(save_path)
                prog_bar.update()
                
        if loss.item() < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'/home/sidd_s/scratch/saved_models/iadb_cond_seg/best_model_parameters.pt')
            print('Model updated! : current best model saved on: ' + str(epoch+1)) 

        torch.save(model.state_dict(), f'/home/sidd_s/scratch/saved_models/iadb_cond_seg/current_model_parameters.pt')
        print('Model updated! : current model saved for epoch: ' + str(epoch+1))        

if __name__ == '__main__':
    main()