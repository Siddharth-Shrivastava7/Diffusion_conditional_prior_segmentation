'''
conditional image segementation map generation, using alpha bending of gaussian and cityscapes gt maps, with conditioned on segformer softmax prediction>>later will replace unet with transformers of ddp
'''
import os
import torch
from torchvision import transforms
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image 
import numpy as np
from  torchvision.transforms.functional import InterpolationMode
import torch.nn.functional as F 
import test_softmax_pred    
from mmcv.cnn import ConvModule
import mmcv 
from tqdm import tqdm
import torch.nn as nn

## latent iadb model
from diffusers import UNet2DModel

## distributed training with DDP 
import torch.multiprocessing as mp 
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group, destroy_process_group  

## image feature extractor
from dino_mod import ViTExtractor

## semantic label map autoencoder
from semantic_map_autoencoder import Myautoencoder

## for converting ids to train_ids and train_ids to color images 
from cityscapesscripts.helpers.labels import labels

torch.backends.cudnn.benchmark = True ## for better speed ## trying without this ## for CNN specific

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

def ddp_setup(rank: int, world_size: int): 
    """
    Args:
        rank: Unique identifier of each process(gpu)
        world_size: Total number of processes(gpus)
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1" ## master since only one machine(node) we are using which have multiple processes (gpus) in it
    os.environ["MASTER_PORT"] = "29501" ## change the port, when one port is filled up
    init_process_group(backend="nccl", rank=rank, world_size=world_size) # initializes the distributed process group.
    torch.cuda.set_device(rank) # sets the default GPU for each process. This is important to prevent hangs or excessive memory utilization on GPU:0


## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self, root_dir, suffix,  resize_shape, mode = 'train'):
        self.root_dir = root_dir
        self.label_list = []
        self.img_list = []
        self.pred_list = []
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
                            self.pred_list.append(os.path.join(self.pred_dir, name.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')))
                            self.img_list.append(os.path.join(self.img_dir, name.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')))
                    
                ## can't use since in DDP, and distributed sampler is used in Distributed sampler
                # assert len(self.label_list) == len(self.img_list) == len(self.pred_list) == 2975
                

        elif self.mode == 'val': ## dark zurich val  
            self.img_dir = os.path.join(self.root_dir, 'leftImg8bit/dz_val') 
            self.pred_dir = os.path.join(self.root_dir,  'pred/segformerb2/dz_val')
            self.gt_dir = os.path.join(self.root_dir,  'gtFine/dz_val') 
            
            ## TODO from here 
            for path in sorted(os.listdir(self.gt_dir)):
                if path.find(suffix)!=-1:
                    self.gt_list.append(os.path.join(self.gt_dir, path)) 
                    self.pred_list.append(os.path.join(self.pred_dir, path.replace('_gt_labelTrainIds.png', '_rgb_anon.png')))
                    self.img_list.append(os.path.join(self.img_dir, path.replace('_gt_labelTrainIds.png', '_rgb_anon.png')))
            
            ## can't use since in DDP, and distributed sampler is used in Distributed sampler
            # assert len(self.label_list) == len(self.img_list) == len(self.pred_list) == 50

        else: 
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.label_list) 
    
    def __getitem__(self, index):

        img_path = self.img_list[index]
        label_path = self.label_list[index] 
        pred_path = self.pred_list[index]
        img = Image.open(img_path) 
        label = torch.from_numpy(np.array(Image.open(label_path)))
        pred = torch.from_numpy(np.array(Image.open(pred_path)))
        
        ## label_transform is just resizing for both training and validation
        lb_transform = transforms.Compose([ 
        transforms.Resize(self.resize_shape, interpolation=InterpolationMode.NEAREST)
        ])
        label = lb_transform(label.unsqueeze(dim=0))
        
        ## pred transform
        pred = lb_transform(pred.unsqueeze(dim=0)) 
        ## convert pred into RGB  
        pred = label_img_to_color(pred, convert_to_train_id=True)
        ## transformations that could be done in label-rgb image (predictions)
        pred_transform = transforms.Compose([
            transforms.ToTensor()])
        pred = pred_transform(pred)
        
        ## image transform
        if self.mode == 'train':
            img_transform = transforms.Compose([
                transforms.Resize(self.resize_shape),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        elif self.mode == 'val':
            img_transform = transforms.Compose([
                transforms.Resize(self.resize_shape),
                transforms.PILToTensor(),
                transforms.ConvertImageDtype(torch.float),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        img = img_transform(img)
            
        return img, label, pred, pred_path


class MyEnsemble(nn.Module): 
    def __init__(self, n_channels: int = 3) -> None:
        super().__init__() 
        self.latent_deblending_model = get_model(n_channels) ## latent dimension of semantic labels are of (3x64x64), thus deblending for this dimension
        self.combining_condition_model = ConvModule(
            n_channels*3,
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


def load_train_val_objs(root_dir= "/home/guest/scratch/siddharth/data/dataset/cityscapes/", suffix= "_gtFine_labelTrainIds.png" , ip_latent_channels = 3, resize_shape: tuple = (512, 1024), val_suffix: str = '_gt_labelTrainIds.png'): 

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
        shuffle=False,
        sampler=DistributedSampler(dataset)
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
                gpu_id: int, 
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
        self.model = DDP(self.model, device_ids=[self.gpu_id])  ## this is how to wrap model around DDP   
        self.save_imgs_dir = save_imgs_dir
        self.nb_steps = nb_steps
        self.semantic_autoencoder_checkpoint_dir = semantic_autoencoder_checkpoint_dir
        
        ## image feats
        self.img_encoder = ViTExtractor("dino_vits8", stride=8, device=self.gpu_id)  ## for image encoding part (the number of channels is fixed for now, later need to undo hardcode>>the num channels is 384) ## this extracted features will concatenated in the 2nd level of latent iadb UNet
        
        ## latent semantic map feats
        ## semantic label map autoencoder ## loading the current pretrained model
        self.semantic_map_autoencoder = Myautoencoder(in_channels=3, out_channels=self.num_classes) 
        semantic_checkpoint = torch.load(os.path.join(self.semantic_autoencoder_checkpoint_dir, 'current_checkpoint.pt'), map_location=torch.device('cuda:' + str(self.gpu_id)))
        self.semantic_map_autoencoder.load_state_dict(semantic_checkpoint) ## the recommended way (given by pytorch) of loading models!
        self.semantic_map_autoencoder.eval()
        
        ## reducing the dimensionity of the image feats using 1x1 conv and upsampling by learnable params
        self.reduce_image_dim = ConvModule(
            384,
            3,
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ).to(self.gpu_id) 
        self.upsample_image_feats = nn.ConvTranspose2d(in_channels= 3, out_channels= 3, kernel_size=2, stride=2).to(self.gpu_id) ## from (3,32,32) => (3,64,64)
    
    def _run_batch(self, conditional_feats, alphas, mask, targets):
        self.optimizer.zero_grad()
        output = self.model(conditional_feats, alphas)
        loss = torch.sum((output - targets)[mask]**2) # targets as (x1-x0)  
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0]) # batch size 
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        for img, label, pred, _ in self.train_data:

            #creating mask where 1 is for non-ignored labels, 0 for ignored labels
            mask = (label != 255) # B, 1, H, W 
            mask = mask.repeat(1, self.num_classes, 1, 1) # B, num_classes, H, W 

            ## random foreground class label for background in GTs
            label[label == 255] = torch.randint(0, self.num_classes, (1,)).item()  
            
            ## label id to color (RGB)
            label_color = label_img_to_color(label, convert_to_train_id=True)
            
            ## >>x1 being the target latent distribution<< 
            x1 = self.semantic_map_autoencoder.encode(label_color.to(self.gpu_id)) 
            # x1 = x1.permute(0,3,1,2).to(self.gpu_id) 

            ## x0 being the stationary distribution! 
            x0 = torch.randn_like(x1.float()) 

            ## similar to what IADB have defined
            targets = (x1 - x0)

            ## alpha blending taking place between x0 (not conditional feats!) and x1 
            alphas = torch.rand(b_sz).to(self.gpu_id)
            x_alphas = alphas.view(-1,1,1,1) * x1 + (1-alphas).view(-1,1,1,1) * x0 
            
            ## conditional feats => {latent semantic map pred,  x_alphas(latent_semantic_map_gt, stationary gaussian, alphas), image feats}
            self.img_feats = self.img_encoder.extract_descriptors(img.float().to(self.gpu_id)) # B,384,32,32 
            self.img_feats = self.reduce_image_dim(self.img_feats) ## B,3,32,32
            self.img_feats = self.upsample_image_feats(self.img_feats)  ## B,3,64,64
            self.latent_semantic_map_pred = self.semantic_map_autoencoder.encode(pred) # B,3,64,64

            ## similar to DDP -- condition input 
            conditional_feats = torch.cat([self.latent_semantic_map_pred, x_alphas, self.img_feats], dim=1)  
            self._run_batch(conditional_feats, alphas, mask, targets) 

            
    def _save_checkpoint(self, epoch, save_best = False):
        checkpoint = self.model.module.state_dict()
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


    def _sample_conditional_seg_iadb(self, img, label, pred):
        with torch.no_grad(): 
            self.model = self.model.eval()
            
            ## conditional feats => {latent semantic map pred,  x_alphas(latent_target_trained_model, stationary gaussian, alphas), image feats} 
            self.img_feats = self.img_encoder.extract_descriptors(img.float().to(self.gpu_id)) # B,384,32,32 
            self.img_feats = self.reduce_image_dim(self.img_feats) ## B,3,32,32
            self.img_feats = self.upsample_image_feats(self.img_feats)  ## B,3,64,64
            self.latent_semantic_map_pred = self.semantic_map_autoencoder.encode(pred) # B,3,64,64
            
            ## x0 as the stationary distribution 
            x0 = torch.randn_like(self.latent_semantic_map_pred) ## sort of logits 
            
            
            ## now deblending starts: 
            x_alpha = x0 # our stationary distribution is Gaussian distribution only! 
            for t in tqdm(range(self.nb_steps)):
                alpha_start = (t/self.nb_steps)
                alpha_end =((t+1)/self.nb_steps)

                ## conditional input 
                conditional_feats = torch.cat([self.latent_semantic_map_pred, x_alpha, self.img_feats], dim=1)
            
                ## this is giving ~ (\bar_{x1} - \bar{x0})
                d = self.model(conditional_feats, torch.as_tensor(alpha_start, device=self.gpu_id)) 
                
                ## reaching x1 by finding neighbouring x_alphas
                x_alpha = x_alpha + (alpha_end-alpha_start)*d

            x_alpha_decoded = self.semantic_map_autoencoder.decode(x_alpha) ### decoder to output: (B,19,256,256)
            approx_x1_sample_softmax = F.softmax(x_alpha_decoded, dim=1)
            approx_x1_sample = torch.argmax(approx_x1_sample_softmax, dim=1)
            ## for the loss :: between label (x1) and approximated x1 through x_alpha

            val_batch_loss = F.cross_entropy(x_alpha, label.to(self.gpu_id).long().squeeze(dim=1), ignore_index=255) ## as the cross-entropy cares about the order of the discrete ground truth labels 

            return approx_x1_sample, val_batch_loss

        

    def _run_val_sampling(self, epoch):
        print('In Sampling at epoch:' + str(epoch)) 
        save_imgs_dir_ep = os.path.join(self.save_imgs_dir, 'mask_loss_' + str(epoch))
        if not os.path.exists(save_imgs_dir_ep):
            os.makedirs(save_imgs_dir_ep) 
        
        val_b_sz = len(next(iter(self.val_data))[0]) # val batch size ## here taking it as 1
        self.val_data.sampler.set_epoch(epoch)
        val_epoch_loss = 0.0  # Initialize the cumulative loss for the epoch
        prog_bar = mmcv.ProgressBar(len(self.val_data))
        for img, label, pred, pred_path in self.val_data:
            approx_x1_sample, val_batch_loss = self._sample_conditional_seg_iadb(img, label, pred)  
            save_path = os.path.join(save_imgs_dir_ep, pred_path[0].split('/')[-1].replace('_rgb_anon.png', '_predFine_color.png'))
            approx_x1_sample_color = Image.fromarray(label_img_to_color(approx_x1_sample.detach().cpu()))
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
                if self.gpu_id == 0: ## for gpu_id = 0, since We only need to save one model copy of the orignial model
                    self._save_checkpoint(epoch)
                    print('Model updated! : current model saved for epoch: ' + str(epoch))
                    if val_average_loss < self.best_loss:
                        self.best_loss = val_average_loss 
                        self._save_checkpoint(epoch, save_best=True)
                        print('Model updated! : current best model saved on: ' + str(epoch)) 
                

def main(rank: int, world_size: int, save_every: int, total_epochs: int, nb_steps: int, num_classes: int, save_imgs_dir: str, root_dir: str, suffix: str , checkpoint_dir: str, batch_size: int, resize_shape: tuple, semantic_autoencoder_checkpoint_dir: str, val_suffix: str, ip_latent_channels: int):
  
    ddp_setup(rank, world_size) 
    train_set, val_set, model, optimizer = load_train_val_objs(root_dir, suffix, ip_latent_channels, resize_shape, val_suffix)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size=1) ## taking batch size for val equal to 1 

    trainer = Trainer( 
        model, train_data, val_data, optimizer, save_every, checkpoint_dir, num_classes, save_imgs_dir, nb_steps, rank, semantic_autoencoder_checkpoint_dir
    )
    trainer.train(total_epochs)
    destroy_process_group()

if __name__ == '__main__':
    resize_shape = (256, 256) ## testing with lower dimension, for checking its working
    save_every = 25
    total_epochs = 860 ## similar to DDP 160k iter @ batch size 16
    nb_steps = 256 ## similar to IADB ## increasing cause in latent dimension >> can increase more
    num_classes = 19 ## only considering foreground labels 
    save_imgs_dir = '/home/guest/scratch/siddharth/data/results/latent_mask_loss_iadb_cond_seg/result_val_images'
    root_dir= "/home/guest/scratch/siddharth/data/dataset/cityscapes/"
    suffix = '_gtFine_labelTrainIds.png'
    val_suffix = '_gt_labelTrainIds.png'
    batch_size = 12
    checkpoint_dir = '/home/guest/scratch/siddharth/data/saved_models/latent_mask_loss_iadb_cond_seg/' 
    semantic_autoencoder_checkpoint_dir = '/home/guest/scratch/siddharth/data/saved_models/semantic_map_autoencoder/dz_val'
    ip_latent_channels = 3
    
    # Include new arguments rank (replacing device) and world_size. ## rank is auto-allocated by DDP when calling mp.spawn. ### world_size is the number of processes across the training job. For GPU training, this corresponds to the number of GPUs in use, and each process works on a dedicated GPU.
    world_size = torch.cuda.device_count()
    print('world size is: ', world_size)   
        
    mp.spawn(main, args=(world_size, save_every, total_epochs, nb_steps, num_classes, save_imgs_dir, root_dir, suffix, checkpoint_dir, batch_size, resize_shape, semantic_autoencoder_checkpoint_dir, val_suffix,ip_latent_channels), nprocs=world_size)
