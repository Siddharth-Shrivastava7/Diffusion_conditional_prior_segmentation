'''
    semantic map auto encoder :: for finding the latent dimension of semantic segmentation maps 
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
import mmcv 
from tqdm import tqdm
import torch.nn as nn

## for converting ids to train_ids and train_ids to color images 
from cityscapesscripts.helpers.labels import labels

torch.backends.cudnn.benchmark = True ## for better speed ## trying without this ## for CNN specific

from semanticmodules import Encoder, Decoder, DiagonalGaussianDistribution



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

## building custom dataset for auto-encoding process 
class custom_cityscapes_labels(Dataset):
    def __init__(self, root_folder: str = '/home/sit/phd/anz208849/scratch/data/dataset/cityscapes/', pred_dir: str = 'pred/segformerb2', gt_dir: str = 'gtFine', img_dir:str = 'leftImg8bit' , suffix: str = '_labelTrainIds.png' , mode: str = 'train', lb_transform = None, img_transform = None): 
        
        self.img_list = [] 
        self.pred_list = []
        self.gt_list = []
        self.lb_transform = lb_transform 
        self.img_transform = img_transform
    
        if mode == 'train': ## perturbed cityscapes (with random erasing) 
            self.img_dir = os.path.join(root_folder, img_dir, 'custom_train') 
            self.pred_dir = os.path.join(root_folder, pred_dir, 'custom_train') 
            self.gt_dir = os.path.join(root_folder, gt_dir, 'train') 

            for root, dirs, files in os.walk(self.gt_dir, topdown=False):
                for name in tqdm(sorted(files)):
                    path = os.path.join(root, name)
                    if path.find(suffix)!=-1:
                        self.gt_list.append(path)   
                        self.pred_list.append(os.path.join(self.pred_dir, name.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')))
                        self.img_list.append(os.path.join(self.img_dir, name.replace('_gtFine_labelTrainIds.png', '_leftImg8bit.png')))
            
        elif mode == 'val': ## darkzurich val images (never seen by segformerb2)
            self.img_dir = os.path.join(root_folder, img_dir, 'dz_val') 
            self.pred_dir = os.path.join(root_folder, pred_dir, 'dz_val')
            self.gt_dir = os.path.join(root_folder, gt_dir, 'dz_val') 

            for path in sorted(os.listdir(self.gt_dir)):
                if path.find(suffix)!=-1:
                    self.gt_list.append(os.path.join(self.gt_dir, path)) 
                    self.pred_list.append(os.path.join(self.pred_dir, path.replace('_gt_labelTrainIds.png', '_rgb_anon.png')))
                    self.img_list.append(os.path.join(self.img_dir, path.replace('_gt_labelTrainIds.png', '_rgb_anon.png'))) 

        if mode == 'train':
            assert len(self.gt_list) == len(self.img_list) == len(self.pred_list) == 2975
        elif mode == 'val':
            assert len(self.gt_list) == len(self.img_list) == len(self.pred_list) == 50
        else:
            raise Exception('mode has to be either train or val')

    def __len__(self):
        return len(self.gt_list) 
    
    def __getitem__(self, index):

        ## for auto-encoding only gt and pred labels to be used 

        ## gt is discrete label map, to be used in CE loss 
        gt_path = self.gt_list[index]  
        gt = torch.from_numpy(np.array(Image.open(gt_path))) 

        ## currently pred is discrete single channel image but it is required to be in color RGB version!
        pred_path = self.pred_list[index] 
        pred = torch.from_numpy(np.array(Image.open(pred_path)))   

        if self.lb_transform: ## resizing similar to resolution at which segformer was trained  
            gt = self.lb_transform(gt.unsqueeze(dim=0))
            pred = self.lb_transform(pred.unsqueeze(dim=0)) 
        
        ## convert pred into RGB  
        pred = label_img_to_color(pred, convert_to_train_id=True)
        if self.img_transform:
            pred = self.img_transform(pred)
        
        return pred, gt, pred_path


class Myautoencoder(nn.Module):  ## inspired from latent diffusion model paper
    def __init__(self, in_channels: int = 3, out_channels: int = 19) -> None:
        super().__init__() 
        self.z_channels = 3
        self.embed_dim = 3
        self.encoder = Encoder(z_channels=self.z_channels, num_res_blocks=2, in_channels=in_channels, attn_resolutions= [ ], ch=128, ch_mult=[1,2,4], dropout=0.0, double_z=True, out_ch=out_channels, resolution=256) 
        self.decoder = Decoder(out_ch=out_channels, resolution=256,z_channels=3 ,num_res_blocks=2, attn_resolutions=[ ], dropout=0.0, ch=128, ch_mult=[1,2,4], in_channels=in_channels)
        self.quant_conv = torch.nn.Conv2d(2*self.z_channels, 2*self.embed_dim,1) 
        self.post_quant_conv = torch.nn.Conv2d(self.embed_dim, self.z_channels, 1)

    def encode(self, x):
        h = self.encoder(x) 
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments) 
        return posterior 

    def decode(self, z):
        z = self.post_quant_conv(z) 
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior = True): 
        posterior = self.encode(input) 
        if sample_posterior:
            z = posterior.sample() 
        else: 
            z = posterior.mode() 
        dec = self.decode(z)
        return dec, posterior


def load_train_val_objs(root_folder: str = '/home/sit/phd/anz208849/scratch/data/dataset/cityscapes/', pred_dir: str = 'pred/segformerb2', gt_dir: str = 'gtFine', img_dir:str = 'leftImg8bit' , suffix: str = '_gtFine_labelTrainIds.png' , num_classes = 19, resize_shape: tuple = (1024, 1024), resume_from: bool= False, checkpoint_dir: str = '/home/sit/phd/anz208849/scratch/data/saved_models/semantic_map_autoencoder/dz_val'): 

    ## transforms for gt and predictions 
    lb_transform = transforms.Compose([ 
        transforms.Resize(resize_shape, interpolation=InterpolationMode.NEAREST)
    ])
    ## transformations that could be done in label-rgb image (predictions)
    img_transform = transforms.Compose([
        transforms.ToTensor() 
    ])

    ## datasets from
    train_set = custom_cityscapes_labels(root_folder, pred_dir, gt_dir, img_dir, suffix, mode='train', lb_transform=lb_transform, img_transform=img_transform)
    val_set = custom_cityscapes_labels(root_folder, pred_dir, gt_dir, img_dir, suffix, mode='val', lb_transform=lb_transform, img_transform=img_transform)

    model = Myautoencoder(in_channels=3, out_channels=num_classes) 
    if resume_from:
        print('loading previous trained model and starting training from there')
        checkpoint = torch.load(os.path.join(checkpoint_dir, 'current_checkpoint.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    return train_set, val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int): ## to set number of workers here 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4 ## use when not using DDP based training 
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
                device: torch.device) -> None: 
        
        self.gpu_id = device
        self.model = model.to(self.gpu_id) 
        self.train_data = train_data 
        self.val_data = val_data 
        self.optimizer = optimizer
        self.save_every = save_every 
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = num_classes
        self.best_loss = torch.finfo(torch.float32).max # init the best loss 
        self.save_imgs_dir = save_imgs_dir
              

    def _run_batch(self, pred, target):
        self.optimizer.zero_grad()
        output, posterior = self.model(pred.to(self.gpu_id)) ## 19 channel logits for calculating CE loss  
        loss = F.cross_entropy(output, target.to(self.gpu_id).long().squeeze(dim=1), ignore_index=255)
        loss.backward()
        self.optimizer.step()
    
    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0]) # batch size 
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        for pred, gt, _ in tqdm(self.train_data):
            self._run_batch(pred=pred, target=gt) 

    def _save_checkpoint(self, epoch, save_best = False):
        checkpoint = self.model.state_dict()
        if save_best:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pt')
            torch.save(checkpoint.state_dict(), checkpoint_path)
            print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}") 
        else:
            checkpoint_path = os.path.join(self.checkpoint_dir, 'current_checkpoint.pt')
            torch.save(checkpoint.state_dict(), checkpoint_path)
            print(f"Epoch {epoch} | Training checkpoint saved at {checkpoint_path}")


    def _run_val_sampling(self, epoch):
        print('Inference at epoch:' + str(epoch)) 
        save_imgs_dir_ep = os.path.join(self.save_imgs_dir, str(epoch))
        if not os.path.exists(save_imgs_dir_ep):
            os.makedirs(save_imgs_dir_ep) 
        
        val_b_sz = len(next(iter(self.val_data))[0]) # val batch size ## here taking it as 1
        val_epoch_loss = 0.0  # Initialize the cumulative loss for the epoch
        prog_bar = mmcv.ProgressBar(len(self.val_data))
        for pred, gt, pred_path in self.val_data:
            with torch.no_grad(): 
                self.model = self.model.eval()
                output, posterior = self.model(pred.to(self.gpu_id))
                val_batch_loss = F.cross_entropy(output, gt.to(self.gpu_id).long().squeeze(dim=1), ignore_index=255)
                
            output_softmax =  F.softmax(output, dim=1)
            output_sample = torch.argmax(output_softmax, dim=1)
            save_path = os.path.join(save_imgs_dir_ep, pred_path[0].split('/')[-1])
            approx_x1_sample_color = Image.fromarray(label_img_to_color(output_sample.detach().cpu()))
            approx_x1_sample_color.save(save_path)
            # Accumulate batch loss to epoch loss
            val_epoch_loss += val_batch_loss.item()
            prog_bar.update()

        # Calculate average loss for the epoch
        val_average_loss = val_epoch_loss / val_b_sz  # Number of batches in the epoch
        return val_average_loss 
        

    def train(self, max_epochs: int):
        for epoch in tqdm(range(max_epochs)):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch)
                print('Model updated! : current model saved for epoch: ' + str(epoch))

                val_average_loss =  self._run_val_sampling(epoch)
                if val_average_loss < self.best_loss:
                    self.best_loss = val_average_loss 
                    self._save_checkpoint(epoch, save_best=True)
                    print('Model updated! : current best model saved on: ' + str(epoch)) 
                

def main():
    resize_shape = (256, 256)  # once for testing latent diffusion code then later will need to change to (1024, 1024) dimension or any other higher resolution
    save_every = 25
    total_epochs = 860 ## similar to DDP 160k iter @ batch size 16
    num_classes = 19 ## only considering foreground labels 
    save_imgs_dir = '/home/sit/phd/anz208849/scratch/data/results/semantic_map_autoencoder/dz_val'
    root_folder = '/home/sit/phd/anz208849/scratch/data/dataset/cityscapes/'
    pred_dir = 'pred/segformerb2'
    gt_dir = 'gtFine'
    suffix = '_labelTrainIds.png'
    img_dir = 'leftImg8bit' 
    batch_size = 8 ## batch size 12 in latent diffusion model, but here getting out of memory, so reducing for now, later will try to make it 12
    checkpoint_dir = '/home/sit/phd/anz208849/scratch/data/saved_models/semantic_map_autoencoder/dz_val' 
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 

    train_set, val_set, model, optimizer = load_train_val_objs(root_folder, pred_dir, gt_dir, img_dir, suffix, num_classes, resize_shape)
    train_data = prepare_dataloader(train_set, batch_size)
    val_data = prepare_dataloader(val_set, batch_size=1) ## taking batch size for val equal to 1 

    trainer = Trainer( 
        model, train_data, val_data, optimizer, save_every, checkpoint_dir, num_classes, save_imgs_dir, device
    )
    trainer.train(total_epochs)


if __name__ == '__main__':
    main()
    

    
