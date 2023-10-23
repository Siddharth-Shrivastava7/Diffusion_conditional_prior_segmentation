'''
conditional image segementation map generation, using alpha bending of gaussian and cityscapes gt maps, with conditioned on segformer softmax prediction>>later will replace unet with transformers of ddp
'''

import torch
from torchvision import transforms
from diffusers import UNet2DModel
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
import os
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

## distributed training with torchrun (fault tolerance with elasticity) 
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

torch.backends.cudnn.benchmark = True ## for better speed 
os.environ["LOCAL_RANK"] = "0" ## requires torchrun ## have to see, how to set this 
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 3, 4, 5" ## may be tricky to use here, still taking risk

def ddp_setup():
    init_process_group(backend="nccl")
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"])) ## trying cuda visible devices environment variable, as suggested in pytorch docs 


def get_model(num_classes):
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

## building custom dataset for x1 of alpha blending procedure 
class custom_cityscapes_labels(Dataset):
    def __init__(self,gt_dir, suffix, lb_transform = None, mode = 'train'):
        suffix = '_gtFine_labelTrainIds.png' 
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
            label = self.lb_transform(label.unsqueeze(dim=0))
        
        return label, img_path


class MyEnsemble(nn.Module): 
    def __init__(self, embed_dim) -> None:
        super().__init__() 
        self.denoising_model = get_model(embed_dim)
        self.combining_condition_model = ConvModule(
            embed_dim * 2,
            embed_dim,
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        )
        
    def forward(self, conditional_feats, alphas, in_val = False):
        conditional_feats = self.combining_condition_model(conditional_feats)
        if in_val:
            d = self.denoising_model(conditional_feats, alphas)['sample']
        else:
            d = self.denoising_model(conditional_feats, alphas)
        return d 


def load_train_val_objs(gt_dir= "/home/guest/scratch/siddharth/data/dataset/cityscapes/gtFine/", suffix= "_gtFine_labelTrainIds.png" , num_classes = 19, resize_shape: tuple = (512, 1024)): 

    lb_transform = transforms.Compose([ 
        transforms.Resize(resize_shape, interpolation=InterpolationMode.NEAREST)
    ])
    train_set =  custom_cityscapes_labels(gt_dir, suffix, lb_transform, mode='train')# loading training dataset
    val_set = custom_cityscapes_labels(gt_dir, suffix, lb_transform, mode = 'val')
    
    model = MyEnsemble(embed_dim=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int): ## to set number of workers here 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4, ## default case : for each gpu 4 is the standard num_workers to use
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
                snapshot_dir: str,
                _to_correct_model_path: str, 
                _to_correct_config_path: str, 
                num_classes: int, 
                save_imgs_dir: str, 
                nb_steps: int) -> None: 
        
        self.gpu_id = int(os.environ["LOCAL_RANK"]) 
        self.model = model.to(self.gpu_id) 
        self.train_data = train_data 
        self.val_data = val_data 
        self.optimizer = optimizer
        self.save_every = save_every 
        self.snapshot_dir = snapshot_dir
        self.epochs_run = 0
        self.softmax_logits_to_correct_train, self.softmax_logits_to_correct_val = self._softmax_logits_predictions(_to_correct_model_path, _to_correct_config_path)  
        self.num_classes = num_classes
        self.best_loss = torch.finfo(torch.float32).max # init the best loss 
        
        if os.path.exists(os.path.join(snapshot_dir, 'current_snapshot.pt')):
            print("Loading snapshot")
            self._load_snapshot(snapshot_dir) 
        
        self.model = DDP(self.model, device_ids=[self.gpu_id])  ## this is how to wrap model around DDP   
        self.save_imgs_dir = save_imgs_dir
        self.nb_steps = nb_steps

    def _softmax_logits_predictions(self, model_path, config_path):
        results_softmax_predictions_train, results_softmax_predictions_val = test_softmax_pred.main(config_path=config_path, checkpoint_path= model_path)
        print('results consisting of softmax predictions loaded successfully!')
        return results_softmax_predictions_train, results_softmax_predictions_val
    
    def _load_snapshot(self, snapshot_dir): 
        loc = f"cuda:{self.gpu_id}"
        snapshot_path = os.path.join(snapshot_dir, 'current_snapshot.pt')
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

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
        for gt_labels, img_paths in self.train_data:

            #creating mask where 1 is for non-ignored labels, 0 for ignored labels
            mask = (gt_labels != 255) # B, 1, H, W 
            mask = mask.repeat(1, self.num_classes, 1, 1) # B, num_classes, H, W 

            ## random foreground class label for background in GTs
            gt_labels[gt_labels == 255] = torch.randint(0, self.num_classes, (1,)).item() 
            
            ## >>x1 being the target distribution<< 
            x1 = F.one_hot(gt_labels.squeeze().long(), self.num_classes) # consist only foreground labels
            x1 = x1.permute(0,3,1,2).to(self.gpu_id) 

            ## x0 being the stationary distribution! 
            x0 = torch.randn_like(x1.float()) 

            ## similar to what IADB have defined
            targets = (x1 - x0)

            ## alpha blending taking place between x0 (not conditional feats!) and x1 
            alphas = torch.rand(b_sz).to(self.gpu_id)
            x_alphas = alphas.view(-1,1,1,1) * x1 + (1-alphas).view(-1,1,1,1) * x0 
            
            ## our way :: to use pre-defined conditioning :: segformerb2 softmax-logits
            pred_labels_emdb  = [torch.tensor(self.softmax_logits_to_correct_train[path]) for path in img_paths] 
            pred_labels_emdb = torch.stack(pred_labels_emdb).to(self.gpu_id) # B,C,H,W ## here C = 19   

            ## similar to DDP -- condition input 
            conditional_feats = torch.cat([pred_labels_emdb, x_alphas], dim=1)  
            self._run_batch(conditional_feats, alphas, mask, targets) 

            
    def _save_snapshot(self, epoch, save_best = False):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        if save_best:
            snapshot_path = os.path.join(self.snapshot_dir, 'best_snapshot.pt')
            torch.save(snapshot, snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}") 
        else:
            snapshot_path = os.path.join(self.snapshot_dir, 'current_snapshot.pt')
            torch.save(snapshot, snapshot_path)
            print(f"Epoch {epoch} | Training snapshot saved at {snapshot_path}")


    def _sample_conditional_seg_iadb(self, gt_label, img_path):
        with torch.no_grad(): 
            self.model = self.model.eval()

            ## predictions of segformerb2 as the conditions
            pred_labels_emdb  = [torch.tensor(self.softmax_logits_to_correct_val[path]) for path in img_path] # conditioning softmax prediciton
            pred_labels_emdb = torch.stack(pred_labels_emdb).to(self.gpu_id) # B,C,H,W ## here C = 19    

            ## x0 as the stationary distribution 
            x0 = torch.randn_like(pred_labels_emdb) ## sort of logits 
            
            ## now deblending starts: 
            x_alpha = x0 # our stationary distribution is Gaussian distribution only! 
            for t in tqdm(range(self.nb_steps)):
                alpha_start = (t/self.nb_steps)
                alpha_end =((t+1)/self.nb_steps)

                ## conditional input 
                conditional_feats = torch.cat([pred_labels_emdb, x_alpha], dim=1)
            
                ## this is giving ~ (\bar_{x1} - \bar{x0})
                d = self.model(conditional_feats, torch.as_tensor(alpha_start, device=self.gpu_id), in_val = True) 
                
                ## reaching x1 by finding neighbouring x_alphas
                x_alpha = x_alpha + (alpha_end-alpha_start)*d

            approx_x1_sample_softmax = F.softmax(x_alpha, dim=1)
            approx_x1_sample = torch.argmax(approx_x1_sample_softmax, dim=1)
            ## for the loss :: between gt_label (x1) and approximated x1 through x_alpha

            val_batch_loss = F.cross_entropy(x_alpha, gt_label, ignore_index=255) ## as the cross-entropy cares about the order of the discrete ground truth labels 

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
        for gt_label, img_path in self.val_data:
            approx_x1_sample, val_batch_loss = self._sample_conditional_seg_iadb(gt_label, img_path)  
            save_path = os.path.join(save_imgs_dir_ep, img_path.split('/')[-1].replace('_leftImg8bit.png', '_predFine_color.png'))
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
            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
                print('Model updated! : current model saved for epoch: ' + str(epoch))

                val_average_loss =  self._run_val_sampling(epoch)
                if val_average_loss < self.best_loss:
                    self.best_loss = val_average_loss 
                    self._save_snapshot(epoch, save_best=True)
                    print('Model updated! : current best model saved on: ' + str(epoch)) 
                


def main(to_correct_model_path: str, to_correct_config_path: str, save_every: int, total_epochs: int, nb_steps: int, num_classes: int, save_imgs_dir: str, gt_dir: str, suffix: str , snapshot_dir: str, batch_size: int=16, resize_shape: tuple = (512, 1024)):
    
    ddp_setup() 
    train_set, val_set, model, optimizer = load_train_val_objs(gt_dir, suffix, num_classes, resize_shape)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size=1) ## taking batch size for val equal to 1 
    trainer = Trainer( 
        model, train_data, val_data, optimizer, save_every, snapshot_dir, 
        to_correct_model_path, to_correct_config_path, num_classes, save_imgs_dir, 
        nb_steps
    )
    trainer.train(total_epochs)
    destroy_process_group()

           

if __name__ == '__main__':
    resize_shape = (512, 1024)
    to_correct_model_path = '/home/guest/scratch/siddharth/data/saved_models/mmseg/segformer_b2_cityscapes_1024x1024/segformer_mit-b2_8x1_1024x1024_160k_cityscapes_20211207_134205-6096669a.pth'
    to_correct_config_path = '/home/guest/scratch/siddharth/data/saved_models/mmseg/segformer_b2_cityscapes_1024x1024/segformer_mit-b2_8xb1-160k_cityscapes-1024x1024.py'
    save_every = 25
    total_epochs = 860 ## similar to DDP 160k iter @ batch size 16
    nb_steps = 128 ## similar to IADB 
    num_classes = 19 ## only considering foreground labels 
    save_imgs_dir = '/home/guest/scratch/siddharth/data/results/mask_loss_iadb_cond_seg/result_val_images'
    gt_dir = '/home/guest/scratch/siddharth/data/dataset/cityscapes/gtFine/'
    suffix = '_gtFine_labelTrainIds.png'
    batch_size = 16
    snapshot_dir = '/home/guest/scratch/siddharth/data/saved_models/mask_loss_iadb_cond_seg/'

    main(to_correct_model_path, to_correct_config_path, save_every, total_epochs, nb_steps, num_classes, save_imgs_dir, gt_dir, suffix, snapshot_dir, batch_size, resize_shape)
