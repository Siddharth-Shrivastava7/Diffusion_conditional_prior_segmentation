
import platform
import os

import numpy as np
from PIL import Image

import torch
from torchvision import transforms, datasets

import ddpm

from datasets.cityscapes_config import encode_target
from typing import Any, Callable, List, Optional, Tuple, Union

BASE_PATH = os.path.expandvars("/home/sidd_s/scratch/dataset/dark_zurich")

NUM_CLASSES = 20
BACKGROUND_CLASS = 19

NORMALIZER = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
COLOR_JITTER = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)


def get_weights():
    return torch.as_tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0])


def labels_to_categories(arr: np.ndarray) -> np.ndarray:
    return encode_target(arr)


def validation_dataset(max_size: int = 64, transforms_dict_val: Union[dict, None] = None):
    # dataset = datasets.Cityscapes(root=BASE_PATH, split="val", mode="fine", target_type="semantic") # change here from cityscpaes to darkzurich   
    # noinspection PyTypeChecker
    dataset = DarkZurich(root=BASE_PATH) # darkzurich dataset
    dataset = ddpm.TransformedImgLblDataset(dataset,
                                            transforms_dict_val,
                                            num_classes=get_num_classes(),
                                            label_mapping_func=labels_to_categories
                                            )

    if max_size:
        dataset, _ = torch.utils.data.random_split(dataset, [max_size, len(dataset) - max_size], generator=torch.Generator().manual_seed(1))
    return dataset


def test_dataset(max_size: int = 128):
    return validation_dataset(max_size)


def get_num_classes() -> int:
    return NUM_CLASSES


def get_ignore_class() -> int:
    return BACKGROUND_CLASS

def train_ids_to_class_names(): 
    ids_to_cls = {
        0: 'road', 
        1: 'sidewalk',
        2: 'building', 
        3: 'wall', 
        4: 'fence', 
        5: 'pole', 
        6: 'traffic light', 
        7: 'traffic sign',
        8: 'vegetation', 
        9: 'terrain', 
        10: 'sky', 
        11: 'person', 
        12: 'rider', 
        13: 'car', 
        14: 'truck', 
        15: 'bus', 
        16: 'train', 
        17: 'motorcycle', 
        18: 'bicycle', 
        19: 'background' 
    }
    return ids_to_cls


class DarkZurich(datasets.Cityscapes):
    def __init__(self,  
                root: str
                ): 
        self.root = root
        self.images_dir =  os.path.join(self.root, 'rgb_anon/val/night/GOPR0356')
        self.targerts_dir = os.path.join(self.root, 'gt/val/night/GOPR0356') 
        self.images = []
        self.targets = []
        
        for img in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, img)
            tar = img.replace('/rgb_anon/','/gt/').replace('_rgb_anon.png','_gt_labelIds.png') # further it converts into train ids via cityscape encode function
            target_dir = os.path.join(self.targerts_dir, tar)

            self.images.append(img_dir)
            self.targets.append(target_dir)
            
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        
        image = Image.open(self.images[index]).convert('RGB') 
        target = Image.open(self.targets[index])
        
        return image, target