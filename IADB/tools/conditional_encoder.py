from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import logging

from dino_mod import ViTExtractor

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = ["build_cond_encoder"]


class ConditionEncoder(nn.Module):
    def __init__(self):
        super().__init__()


class DummyEncoder(ConditionEncoder):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResNetEncoder(ConditionEncoder):
    def __init__(self, train_encoder: bool, conditioning: str):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet34', pretrained=True)
        self.model.fc = nn.Identity()
        if not train_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
        self.conditioning = conditioning

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.fc(x)
        
        if self.conditioning == 'x-attention':
            x = rearrange(x, 'b f h w -> b (h w) f')
        elif self.conditioning == 'sum':
            x = F.adaptive_avg_pool2d(x, 1).squeeze()

        return x


class DinoViT(ConditionEncoder):
    def __init__(self, name: str,
                 train_encoder: bool,
                 conditioning: str,
                 stride: int = 8,
                 resize_shape: Union[tuple, None] = None,
                 layers: Union[list, int] = 11):
        super().__init__()
        self.extractor = ViTExtractor(name, stride)
        if not train_encoder:
            for param in self.parameters():
                param.requires_grad = False
        self.stride = stride
        self.conditioning = conditioning
        self.layers = layers
        self.resize_shape = resize_shape

    # def parameters(self):
    #     return self.extractor.model.parameters()
    #
    # def eval(self):
    #     self.extractor.model.eval()
    #
    # def train(self):
    #     self.extractor.model.train()

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, list]:
        f = self.extractor.extract_descriptors(x, self.layers, resize_shape=self.resize_shape)
        return f



if __name__ == '__main__':
    a = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # encoder = ResNetEncoder(False, 'x-attention')

    p = {"cond_encoder": "dino_vits8", "dataset_file": "datasets.cityscapes"}

    encoder = ViTExtractor(p["cond_encoder"], stride=8, device="cuda")
    x_ = torch.randn(size=(2, 3, 256, 512))
    # encoder.model(x.float().cuda()) # (2, 384)
    # stride = 8
    descriptors = encoder.extract_descriptors(x_.float().cuda()) # (2, 1, 512, 384) --> torch.Size([2, 384, 32, 64])
    torch.save(descriptors, '/home/guest/scratch/siddharth/models/Diffusion_conditional_prior_segmentation/ccdm-stochastic-segmentation_mod/test.pt')