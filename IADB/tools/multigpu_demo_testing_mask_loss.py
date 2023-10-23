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
from test_softmax_pred import main  ## while debug use abosolute module address 
from mmcv.cnn import ConvModule
import mmcv 
from tqdm import tqdm
import torch.nn as nn

## distributed training with torchrun (fault tolerance with elasticity) 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os


torch.backends.cudnn.benchmark = True ## for better speed 

## condition => the "softmax-logits" prediction of cityscapes dataset from segformer model 
## we will be loading trained model, so the configuration will be that of validation of mmseg model 
segformer_model_path = '/home/guest/scratch/siddharth/data/saved_models/mmseg/segformer_b2_cityscapes_1024x1024/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'
config_file_path = '/home/guest/scratch/siddharth/data/saved_models/mmseg/segformer_b2_cityscapes_1024x1024/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py' 
results_softmax_predictions_train, results_softmax_predictions_val = main(config_path= config_file_path, checkpoint_path= segformer_model_path) # lets check!  ## caching in train and val data softmax predictions; so that segformerb2 not have to predict every other data instant but rather can be easily indexed for softmax prediction generation.
print('results consisting of softmax predictions loaded successfully!')


def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


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
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=num_classes, in_channels=num_classes, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

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
def sample_conditional_seg_iadb(model, datav, conditional_transform, device, nb_step):
    model = model.eval() 
    conditional_transform = conditional_transform.eval() 
    
    ## predictions of segformerb2 as the conditions
    pred_labels_emdb  = [torch.tensor(results_softmax_predictions_val[path]) for path in datav[1]] # conditioning softmax prediciton
    pred_labels_emdb = torch.stack(pred_labels_emdb).to(device) # B,C,H,W ## here C = 19    

    ## x0 as the stationary distribution 
    x0 = torch.randn_like(pred_labels_emdb) ## sort of logits

    ## conditional input 
    conditional_feats = torch.cat([pred_labels_emdb, x0], dim=1)
    conditional_feats = conditional_transform(conditional_feats) ## sort of logits
    
    ## now deblending starts: 
    x_alpha = x0 # our stationary distribution is Gaussian distribution only! 
    for t in tqdm(range(nb_step)):
        alpha_start = (t/nb_step)
        alpha_end =((t+1)/nb_step)

        d = model(conditional_feats, torch.as_tensor(alpha_start, device=x_alpha.device))['sample'] ## this is giving ~ (\bar_{x1} - \bar{x0})
        x_alpha = x_alpha + (alpha_end-alpha_start)*d 

        conditional_feats = torch.cat([pred_labels_emdb, x_alpha], dim=1)
        conditional_feats = conditional_transform(conditional_feats)

    return x_alpha
    

## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self, gt_dir, suffix = '_gtFine_labelTrainIds.png', lb_transform = None, mode = 'train'):
        self.gt_dir = gt_dir + mode  
        self.lb_transform = lb_transform 
        self.label_list = []
        self.img_list = []
        
        for root, dirs, files in os.walk(self.gt_dir, topdown=False):
            for name in tqdm(sorted(files)):
                path = os.path.join(root, name)
                if path.find(suffix)!=-1:
                    self.label_list.append(path) 
                    img_path = path.replace('/gtFine/','/leftImg8bit/').replace('_gtFine_labelTrainIds.png','_leftImg8bit.png')
                    self.img_list.append(img_path)

        if mode == 'train':
            assert len(self.label_list) == 2975 == len(self.img_list)
        elif mode == 'val':
            assert len(self.label_list) == 500 == len(self.img_list)
        else:
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.label_list) 
    
    def __getitem__(self, index):

        img_path = self.img_list[index]

        label_path = self.label_list[index]  
        label = torch.from_numpy(np.array(Image.open(label_path)))
        # label[label==255]=torch.randint(0,19,size=(1,)).item()

        if self.lb_transform: 
            label = self.lb_transform(label.unsqueeze(dim=0)) # resizing the tensor, for working in low dimension
            # pred_label = self.lb_transform(pred_label.unsqueeze(dim=0)) ## Two cases emerge here: 1. resizing segformer output to 128x256, when its input image was 1024x2048 2. resizing the input to 128x256 and then use model prediction resizing at a reduced size, which was earlier in the segformer model was upsampled in a biliner interpolation fashion through its logits. =>> for now, going with 2. way, since we might find improvement earlier here. 
        
        return label, img_path
       

def main(): 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gt_dir = '/home/guest/scratch/siddharth/data/dataset/cityscapes/gtFine/' 
    suffix = "_gtFine_labelTrainIds.png"
    global num_classes
    num_classes = 19 ## only foreground classes 
    embed_dim = num_classes #a hyper-param ##used in order to arrive at a consistency with DDP and overcome the issue of random assignment for background
    # embedding_table = nn.Embedding(num_classes + 1, embedding_dim=embed_dim).to(device) ## not req here, cause using predefined softmax logits of segformerb2 as the conditions
    sampling_epoch_factor = 25 ## after every 25 epochs, perfrom sampling steps 
    # gradient_accumulation_steps = 4 # a hyper-param  ## change the batch statistics, opting to similar batch statics as given in DDP, thus commenting for now. ## from pytorch disscusion forum: Your gradient accumulation approach might change the model performance, if you are using batch-size-dependent layers such as batchnorm layers.
    batch_size = 16 # differ from DDP since compute issuasee
    lb_transform = transforms.Compose([ 
        transforms.Resize((512,1024), interpolation=InterpolationMode.NEAREST), # (H/4, W/4) ## similar to DDP, where they were training using (512, 1024) images and performed diffusion in (H/4, W/4) => (128, 256) and then used bilinear upsampling of the logits to obtain final prediction label.
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
    dataset_train = custom_cityscapes_labels(gt_dir, suffix, lb_transform, mode = 'train')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory = True)  
    ## val dataloader
    dataset_val = custom_cityscapes_labels(gt_dir, suffix, lb_transform, mode = 'val')
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=True, num_workers=4, pin_memory = True)  
    print('dataset loaded successfully!')

    model = get_model() 
    model = model.to(device)
    print('Model loaded into cuda')

    optimizer = Adam((list(model.parameters())  + list(conditional_transform.parameters())), lr=1e-4)  ## optimising multiple models as they are not in the same class 
    
    # hyper-param
    best_loss = torch.finfo(torch.float32).max # init the best loss 
    optimizer.zero_grad() 
    print('Start training')
    for epoch in tqdm(range(860)): ## 860 epochs of cityscapes data which is ~ 160k iterations
        for iter_step, data in tqdm(enumerate(dataloader_train)):
            ## >>x1 being the target distribution<<
            gtlabel = data[0].clone()
            gtlabel[gtlabel == 255] = torch.randint(0,num_classes, (1,)).item() ## random foreground class label for background in GT
            x1 = F.one_hot(gtlabel.squeeze().long(), num_classes) # consist only foreground labels
            x1 = x1.permute(0,3,1,2).to(device)  # B, C, H, W 

            ## x0 being the stationary distribution! 
            x0 = torch.randn_like(x1.float()) # standard normal distribution  # acc to original IADB ## sort of logits
            
            bs = x0.shape[0] # batch size 
            ## alpha blending taking place between x0 (not conditional feats!) and x1 
            alpha = torch.rand(bs, device=device)
            x_alpha = alpha.view(-1,1,1,1) * x1 + (1-alpha).view(-1,1,1,1) * x0

            ## similar to DDP 
            # pred_labels_emdb = embedding_table(data[1].long().to(device, non_blocking=True)).squeeze(1).permute(0, 3, 1, 2) 
            # pred_labels_emdb = (torch.sigmoid(pred_labels_emdb)*2 - 1)*bit_scale ## sort of logits
            ## our way :: to use pre-defined conditioning :: segformerb2 softmax-logits
            pred_labels_emdb  = [torch.tensor(results_softmax_predictions_train[path]) for path in data[1]] # conditioning softmax prediciton
            pred_labels_emdb = torch.stack(pred_labels_emdb).to(device) # B,C,H,W ## here C = 19    
            

            ## condition input 
            conditional_feats = torch.cat([pred_labels_emdb, x_alpha], dim=1)
            conditional_feats = conditional_transform(conditional_feats)
             
            bs = x0.shape[0] # batch size 

            # Create a mask where 1 is for non-ignored labels, 0 for ignored labels
            mask = (data[0] != 255) # B, 1, H, W
            mask = mask.repeat(1, num_classes, 1, 1) # B, num_classes, H, W
            d = model(conditional_feats, alpha)['sample'] ## model involved for denoising/(de-blending here), the blended value.
            # loss = torch.sum((d - (x1-x0))**2) ## based on IADB paper
            loss = torch.sum((d - (x1-x0))[mask]**2) ## based on IADB paper


            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()

            # if (iter_step+1) % gradient_accumulation_steps == 0: ## performing gradient accumulation
            #     optimizer.step()
            #     optimizer.zero_grad()

        if epoch % sampling_epoch_factor == 0:
            print('In Sampling at epoch:' + str(epoch+1))
            dataset = dataloader_val.dataset
            prog_bar = mmcv.ProgressBar(len(dataset))
            save_imgs_dir = '/home/guest/scratch/siddharth/data/results/mask_loss_iadb_cond_seg/result_val_images'
            save_imgs_dir_ep = os.path.join(save_imgs_dir, 'mask_loss_' + str(epoch+1))
            if not os.path.exists(save_imgs_dir_ep):
                os.makedirs(save_imgs_dir_ep)
            for __, datav in enumerate(dataloader_val):
                with torch.no_grad(): 
                    x1_sample_logits = sample_conditional_seg_iadb(model, datav, conditional_transform, device, nb_step=128) ## nb_step is a hyper-param > taken from IADB 
                    x1_sample_softmax = F.softmax(x1_sample_logits, dim=1)
                    argmax_x1_sample = torch.argmax(x1_sample_softmax, dim=1) 
                    save_path = os.path.join(save_imgs_dir_ep, datav[1][0].split('/')[-1].replace('_leftImg8bit.png', '_predFine_color.png'))
                    x1_sample_color = Image.fromarray(label_img_to_color(argmax_x1_sample.detach().cpu()))
                    x1_sample_color.save(save_path)
                    prog_bar.update()
                
        if loss.item() < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), f'/home/guest/scratch/siddharth/data/saved_models/mask_loss_iadb_cond_seg/best_model_parameters.pt')
            print('Model updated! : current best model saved on: ' + str(epoch+1)) 

        torch.save(model.state_dict(), f'/home/guest/scratch/siddharth/data/saved_models/mask_loss_iadb_cond_seg/current_model_parameters.pt')
        print('Model updated! : current model saved for epoch: ' + str(epoch+1))        

if __name__ == '__main__':
    main()
