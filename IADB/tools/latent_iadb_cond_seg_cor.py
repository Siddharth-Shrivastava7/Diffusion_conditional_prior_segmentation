'''
conditional image segementation map generation, using alpha bending of gaussian and cityscapes gt maps, with conditioned on segformer softmax prediction>>later will replace unet with transformers of ddp
'''
import os
import torch
from torchvision import transforms
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import numpy as np
from  torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F 
from mmcv.cnn import ConvModule
import mmcv 
import torch.nn as nn

## latent iadb model
from diffusers import UNet2DModel

## image feature extractor
from dino_mod import ViTExtractor

## semantic label map autoencoder
from semantic_map_autoencoder import Myautoencoder

## for converting ids to train_ids and train_ids to color images 
from cityscapesscripts.helpers.labels import labels

# torch.backends.cudnn.benchmark = True ## for better speed ## trying without this ## for CNN specific

## latent iadb model  
def get_model(n_channels: int = 3):
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
    return UNet2DModel(block_out_channels=block_out_channels,out_channels=n_channels, in_channels=n_channels, up_block_types=up_block_types, down_block_types=down_block_types, add_attention=True)

def label_img_to_color(img, convert_to_train_id = False): 

    if convert_to_train_id: # conversion from id 
            id2train_id = {label.id: label.trainId  for label in labels}    
            for k in range(34): # k is an instance of id
                img[img==k] = id2train_id[k]
        
    ## can also use this below one to form color from train_ids
    # label_to_color = {label.trainId: label.color  for label in labels}  
    # label_to_color[255] = [0,0,0]
    
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
        255: [0,  0, 0] 
        } 
    img = np.array(img.squeeze()) 
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    for row in range(img_height):
        for col in range(img_width):
            label = img[row][col] 
            img_color[row, col] = np.array(label_to_color[label])  
    return img_color


## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self, root_dir, suffix,  resize_shape, mode = 'train'):
        self.root_dir = root_dir
        self.label_list = []
        self.label_latent_list = []
        self.img_enc_list = []
        self.pred_latent_list = []
        self.mode = mode
        self.resize_shape = resize_shape
        
        if self.mode == 'train':
            self.img_dir = os.path.join(self.root_dir, 'leftImg8bit/custom_train') 
            self.pred_dir = os.path.join(self.root_dir, 'pred/segformerb2/custom_train') 
            self.gt_dir = os.path.join(self.root_dir, 'gtFine/train') 
        
            for root, dirs, files in os.walk(self.gt_dir, topdown=False):
                if root.find('/gtFine/train')!= -1:
                    for name in tqdm(sorted(files)):
                        path = os.path.join(root, name)
                        if path.find(suffix)!=-1: # suffix = '_gtFine_labelTrainIds.png'
                            self.label_list.append(path)
                            self.pred_latent_list.append(os.path.join(self.pred_dir, name.replace(suffix,'_leftImg8bit_latent_enc.pt')))
                            self.img_enc_list.append(os.path.join(self.img_dir, name.replace(suffix,'_leftImg8bit_vit_enc.pt')))
                            self.label_latent_list.append(path.replace(suffix, '_gtFine_labelTrainIds_latent_enc.pt'))
                    
                ## can't use since in DDP, and distributed sampler is used in Distributed sampler
                # assert len(self.label_list) == len(self.img_list) == len(self.pred_list) == 2975
                

        elif self.mode == 'val': ## dark zurich val  
            self.img_dir = os.path.join(self.root_dir, 'leftImg8bit/dz_val') 
            self.pred_dir = os.path.join(self.root_dir,  'pred/segformerb2/dz_val')
            self.gt_dir = os.path.join(self.root_dir,  'gtFine/dz_val') 
            
            for path in sorted(os.listdir(self.gt_dir)):
                if path.find(suffix)!=-1:
                    self.label_list.append(os.path.join(self.gt_dir, path)) 
                    self.pred_latent_list.append(os.path.join(self.pred_dir, path.replace(suffix, '_rgb_anon_latent_enc.pt')))
                    self.img_enc_list.append(os.path.join(self.img_dir, path.replace(suffix, '_rgb_anon_latent_vit_enc.pt')))
                    self.label_latent_list.append(os.path.join(self.gt_dir,path.replace(suffix, '_gt_labelTrainIds_latent_enc.pt')))
            
            ## can't use since in DDP, and distributed sampler is used in Distributed sampler
            # assert len(self.label_list) == len(self.img_list) == len(self.pred_list) == 50

        else: 
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.label_list) 
    
    def __getitem__(self, index):

        img_enc_path = self.img_enc_list[index]
        pred_latent_path = self.pred_latent_list[index]
        label_latent_path = self.label_latent_list[index] 
        label_path = self.label_list[index] 
        
        ## loading torch '.pt' files <<offline loading rather than on the fly working on it>> 
        img_enc = torch.load(img_enc_path)
        label_latent = torch.load(label_latent_path) 
        pred_latent = torch.load(pred_latent_path)
        
        ## label loading and transforms 
        label = torch.tensor(np.array(Image.open(label_path)))
        ## label_transform is just resizing for both training and validation
        lb_transform = transforms.Compose([ 
        transforms.Resize(self.resize_shape, interpolation=InterpolationMode.NEAREST)
        ])
        label = lb_transform(label.unsqueeze(dim=0))
        
        return img_enc, label_latent, pred_latent, label, pred_latent_path


class MyEnsemble(nn.Module): 
    def __init__(self, n_channels: int = 3) -> None:
        super().__init__() 
        self.latent_deblending_model = get_model(n_channels) ## latent dimension of semantic labels are of (3x64x64), thus deblending for this dimension
        self.combining_condition_model = ConvModule(
            n_channels*2 + 384, ## since using encoder method in semantic map auto-encoder rather than encode method
            n_channels,
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        
    def forward(self, conditional_feats, alphas):
        conditional_feats = self.combining_condition_model(conditional_feats)
        d = self.latent_deblending_model(conditional_feats, alphas)['sample'] ## sample here is The hidden states output from the last layer of the model. wref: hugging face community
        return d 


def load_train_val_objs(root_dir= "~/scratch/siddharth/data/dataset/cityscapes/", suffix= "_gtFine_labelTrainIds.png" , ip_latent_channels = 3, resize_shape: tuple = (512, 1024), val_suffix: str = '_gt_labelTrainIds.png'): 

    train_set =  custom_cityscapes_labels(root_dir, suffix,  resize_shape, mode='train')# loading training dataset
    val_set = custom_cityscapes_labels(root_dir, val_suffix, resize_shape = resize_shape, mode = 'val')
    
    model = MyEnsemble(ip_latent_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int): ## to set number of workers here 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4, 
        drop_last=True
    )


class Trainer: 
    def __init__(self, 
                model: torch.nn.Module, 
                train_data: DataLoader,
                val_data: DataLoader,
                optimizer: torch.optim.Optimizer,
                save_every: int,
                checkpoint_dir: str,
                num_classes: int, 
                save_imgs_dir: str, 
                nb_steps: int, 
                gpu_id: torch.device, 
                semantic_autoencoder_checkpoint_dir: str) -> None: 
        
        self.gpu_id = gpu_id
        self.model = model.to(self.gpu_id) 
        self.train_data = train_data 
        self.val_data = val_data 
        self.optimizer = optimizer
        self.save_every = save_every 
        self.checkpoint_dir = checkpoint_dir
        self.epochs_run = 0
        self.num_classes = num_classes
        self.best_loss = torch.finfo(torch.float32).max # init the best loss 
        self.save_imgs_dir = save_imgs_dir
        self.nb_steps = nb_steps
        self.semantic_autoencoder_checkpoint_dir = semantic_autoencoder_checkpoint_dir
        
        ## latent semantic map feats
        ## semantic label map autoencoder ## loading the current pretrained model
        self.semantic_map_autoencoder = Myautoencoder(in_channels=3, out_channels=self.num_classes)
        semantic_checkpoint = torch.load(os.path.join(self.semantic_autoencoder_checkpoint_dir, 'current_checkpoint.pt'), map_location=torch.device('cpu'))
        self.semantic_map_autoencoder.load_state_dict(semantic_checkpoint) ## the recommended way (given by pytorch) of loading models!
        self.semantic_map_autoencoder.eval()
    
    def _run_batch(self, conditional_feats, alphas, target):
        self.optimizer.zero_grad()
        output = self.model(conditional_feats, alphas) 
        loss = F.mse_loss(output, target)
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0]) # batch size 
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for img_enc, label_latent, pred_latent, _, _ in tqdm(self.train_data):
            
            ## >>x1 being the target latent distribution<< 
            x1 = label_latent.to(self.gpu_id) 

            ## x0 being the stationary distribution! 
            x0 = torch.randn_like(x1.float()) 

            ## similar to what IADB have defined
            target = (x1 - x0)
            
            ## alpha blending taking place between x0 (not conditional feats!) and x1 
            alphas = torch.rand(b_sz).to(self.gpu_id)
            x_alpha = alphas.view(-1,1,1,1) * x1 + (1-alphas).view(-1,1,1,1) * x0 
            
            ## similar to DDP -- condition input 
            ## conditional feats => {latent semantic map pred,  x_alpha(latent_semantic_map_gt, stationary gaussian, alphas), image feats}
            conditional_feats = torch.cat([pred_latent.to(self.gpu_id) , x_alpha, img_enc.to(self.gpu_id)], dim=1)  
            self._run_batch(conditional_feats, alphas, target) 

            
    def _save_checkpoint(self, epoch, save_best = False):
        checkpoint = self.model.state_dict()
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
        
        if save_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}") 
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'current_checkpoint.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}")


    def _sample_conditional_seg_iadb(self, img_enc, label, pred_latent):
        with torch.no_grad(): 
            self.model = self.model.eval()
            
            ## x0 as the stationary distribution 
            x0 = torch.randn_like(pred_latent.to(self.gpu_id)) ## sort of logits 
            
            ## now deblending starts: 
            x_alpha = x0 # our stationary distribution is Gaussian distribution only! 
            for t in tqdm(range(self.nb_steps)):
                alpha_start = (t/self.nb_steps)
                alpha_end =((t+1)/self.nb_steps)

                ## conditional feats => {latent semantic map pred,  x_alpha(latent_target_trained_model, stationary gaussian, alphas), image feats} 
                conditional_feats = torch.cat([pred_latent.to(self.gpu_id) , x_alpha, img_enc.to(self.gpu_id)], dim=1)
            
                ## this is giving ~ (\bar_{x1} - \bar{x0})
                d = self.model(conditional_feats, torch.as_tensor(alpha_start, device=self.gpu_id)) 
                
                ## reaching x1 by finding neighbouring x_alpha
                x_alpha = x_alpha + (alpha_end-alpha_start)*d

            x_alpha_decoded = self.semantic_map_autoencoder.decode(x_alpha.detach().cpu()) ### decoder to output: (B,19,256,256)
            approx_x1_sample_softmax = F.softmax(x_alpha_decoded, dim=1)
            approx_x1_sample = torch.argmax(approx_x1_sample_softmax, dim=1)
            ## for the loss :: between label (x1) and approximated x1 through x_alpha

            val_batch_loss = F.cross_entropy(x_alpha_decoded, label.long().squeeze(dim=1), ignore_index=255) ## as the cross-entropy cares about the order of the discrete ground truth labels 

            return approx_x1_sample, val_batch_loss

        

    def _run_val_sampling(self, epoch):
        print('In Sampling at epoch:' + str(epoch)) 
        save_imgs_dir_ep = os.path.join(self.save_imgs_dir, 'latent_' + str(epoch))
        if not os.path.exists(save_imgs_dir_ep):
            os.makedirs(save_imgs_dir_ep) 
        
        val_b_sz = len(next(iter(self.val_data))[0]) # val batch size ## here taking it as 1
        val_epoch_loss = 0.0  # Initialize the cumulative loss for the epoch
        prog_bar = mmcv.ProgressBar(len(self.val_data))
        for img_enc, _, pred_latent, label, pred_latent_path in self.val_data:
            approx_x1_sample, val_batch_loss = self._sample_conditional_seg_iadb(img_enc, label, pred_latent)  
            save_path = os.path.join(save_imgs_dir_ep, pred_latent_path[0].split('/')[-1].replace('_rgb_anon_latent_enc.pt', '_predFine_color.png'))
            approx_x1_sample_color = Image.fromarray(label_img_to_color(approx_x1_sample))
            approx_x1_sample_color.save(save_path)
            # Accumulate batch loss to epoch loss
            val_epoch_loss += val_batch_loss.item()
            prog_bar.update()

        # Calculate average loss for the epoch
        val_average_loss = val_epoch_loss / val_b_sz  # Number of batches in the epoch
        return val_average_loss 
        

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0: 
                val_average_loss =  self._run_val_sampling(epoch) ## running validation over all the GPU processes
                self._save_checkpoint(epoch)
                print('Model updated! : current model saved for epoch: ' + str(epoch))
                if val_average_loss < self.best_loss:
                    self.best_loss = val_average_loss 
                    self._save_checkpoint(epoch, save_best=True)
                    print('Model updated! : current best model saved on: ' + str(epoch)) 
                

def main(rank: torch.device, save_every: int, total_epochs: int, nb_steps: int, num_classes: int, save_imgs_dir: str, root_dir: str, suffix: str , checkpoint_dir: str, batch_size: int, resize_shape: tuple, semantic_autoencoder_checkpoint_dir: str, val_suffix: str, ip_latent_channels: int):
  
    train_set, val_set, model, optimizer = load_train_val_objs(root_dir, suffix, ip_latent_channels, resize_shape, val_suffix)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size=1) ## taking batch size for val equal to 1 

    trainer = Trainer( 
        model, train_data, val_data, optimizer, save_every, checkpoint_dir, num_classes, save_imgs_dir, nb_steps, rank, semantic_autoencoder_checkpoint_dir
    )
    trainer.train(total_epochs)

if __name__ == '__main__':
    resize_shape = (256, 256) ## testing with lower dimension, for checking its working
    save_every = 25
    total_epochs = 860 ## similar to DDP 160k iter @ batch size 16
    nb_steps = 256 ## similar to IADB ## increasing cause in latent dimension >> can increase more
    num_classes = 19 ## only considering foreground labels 
    save_imgs_dir = '/home/guest/scratch/siddharth/data/results/latent_iadb_cond_seg_cor/result_val_images'
    root_dir= "/home/guest/scratch/siddharth/data/dataset/cityscapes/"
    suffix = '_gtFine_labelTrainIds.png'
    val_suffix = '_gt_labelTrainIds.png'
    batch_size = 16
    checkpoint_dir = '/home/guest/scratch/siddharth/data/saved_models/latent_iadb_cond_seg_cor/' 
    semantic_autoencoder_checkpoint_dir = '/home/guest/scratch/siddharth/data/saved_models/semantic_map_autoencoder/dz_val'
    ip_latent_channels = 3
    device = torch.device('cuda:5')
    
    main(device, save_every, total_epochs, nb_steps, num_classes, save_imgs_dir, root_dir, suffix, checkpoint_dir, batch_size, resize_shape, semantic_autoencoder_checkpoint_dir, val_suffix,ip_latent_channels)
    