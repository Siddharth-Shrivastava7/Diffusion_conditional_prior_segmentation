import os
import torch
from torchvision import transforms
from tqdm import tqdm 
from torch.utils.data import Dataset, DataLoader
from PIL import Image 
import numpy as np
from  torchvision.transforms.functional import InterpolationMode


## image feature extractor
from dino_mod import ViTExtractor

## semantic label map autoencoder
from semantic_map_autoencoder import Myautoencoder


## for converting ids to train_ids and train_ids to color images 
from cityscapesscripts.helpers.labels import labels

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
            
            for path in sorted(os.listdir(self.gt_dir)):
                if path.find(suffix)!=-1:
                    self.label_list.append(os.path.join(self.gt_dir, path)) 
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
        
        ## label id to color (RGB)
        label_color = label_img_to_color(label, convert_to_train_id=True)
        label_color = pred_transform(label_color)
        
        ## image transform
        img_transform = transforms.Compose([
            transforms.Resize(self.resize_shape),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        img = img_transform(img)
            
        return img, label_color , pred, pred_path, img_path, label_path
    

def prepare_dataloader(dataset: Dataset, batch_size: int): ## to set number of workers here 
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=True,
    )

root_dir= "/home/guest/scratch/siddharth/data/dataset/cityscapes/"
suffix = '_gtFine_labelTrainIds.png'
val_suffix = '_gt_labelTrainIds.png'
resize_shape = (256, 256)

train_set =  custom_cityscapes_labels(root_dir, suffix,  resize_shape, mode='train')# loading training dataset
val_set = custom_cityscapes_labels(root_dir, val_suffix, resize_shape = (256,256), mode = 'val')

train_data = prepare_dataloader(train_set, batch_size=1)
val_data = prepare_dataloader(val_set, batch_size=1)


if __name__ == '__main__':
    ## transforming the input and pred as required (testing phase)
    ## image feats from pretrained vision transformer
    img_encoder = ViTExtractor("dino_vits8", stride=4, device='cpu') 
    ## for image encoding part (the number of channels is fixed for now, later need to undo hardcode>>the num channels is 384)
    
    ## latent semantic map feats
    ## semantic label map autoencoder ## loading the current pretrained model
    semantic_map_autoencoder = Myautoencoder(in_channels=3, out_channels=19)
    semantic_checkpoint = torch.load(os.path.join('/home/guest/scratch/siddharth/data/saved_models/semantic_map_autoencoder/dz_val', 'current_checkpoint.pt'), map_location=torch.device('cpu'))
    semantic_map_autoencoder.load_state_dict(semantic_checkpoint) ## the recommended way (given by pytorch) of loading models!
    semantic_map_autoencoder.eval()
    
    ## saving image encoding and prediction latent features in train dataloader
    
    for img, label_color , pred, pred_path, img_path, label_path in tqdm(train_data):    
        with torch.no_grad():
            img_enc =  img_encoder.extract_descriptors(img.float()) 
            pred_latent = semantic_map_autoencoder.encode(pred).sample() 
            label_color_latent = semantic_map_autoencoder.encode(label_color).sample()
    
        torch.save(img_enc.squeeze(dim=0), img_path[0].replace('_leftImg8bit.png', '_leftImg8bit_vit_enc.pt'))
        torch.save(pred_latent.squeeze(dim=0), pred_path[0].replace('_leftImg8bit.png', '_leftImg8bit_latent_enc.pt'))
        torch.save(label_color_latent.squeeze(dim=0), label_path[0].replace('_gtFine_labelTrainIds.png', '_gtFine_labelTrainIds_latent_enc.pt'))
    
    print('training data saved!')
    
    # for img, label_color , pred, pred_path, img_path, label_path in tqdm(val_data):    
    #     with torch.no_grad():
    #         img_enc =  img_encoder.extract_descriptors(img.float()) 
    #         pred_latent = semantic_map_autoencoder.encode(pred).sample() 
    #         label_color_latent = semantic_map_autoencoder.encode(label_color).sample()

    #     torch.save(img_enc.squeeze(dim=0), img_path[0].replace('_rgb_anon.png', '_rgb_anon_latent_vit_enc.pt'))
    #     torch.save(pred_latent.squeeze(dim=0), pred_path[0].replace('_rgb_anon.png', '_rgb_anon_latent_enc.pt'))
    #     torch.save(label_color_latent.squeeze(dim=0), label_path[0].replace('_gt_labelTrainIds.png', '_gt_labelTrainIds_latent_enc.pt'))
        
    # print('validation data saved!')