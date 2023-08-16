'''
Self Similarity based Diffusion model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from torch.special import expm1
from einops import rearrange, reduce, repeat
from mmcv.cnn import ConvModule
import math
from PIL import Image
import numpy as np
import os

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from ..discrete_diffusion.schedule_mod import q_mats_from_onestepsdot, q_pred_from_mats


@SEGMENTORS.register_module()
class SSD(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                timesteps = 20, 
                **kwargs):
        super(SSD, self).__init__(**kwargs)
        
        self.timesteps = timesteps 
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.num_classes + 1) # instead of one hot encoding making class embedding module for discrete data space
        self.transform = ConvModule(
            self.decode_head.in_channels[0] + (self.num_classes + 1),
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ) # used for converting concatenated encoded i/p image and encoded;corrupted gt map to feature maps of dimension being half of the joint dimension of the concatenated inputs
        
        
        
    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.
        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)[0]  # bs, 256, h/4, w/4
        batch, c, h, w, device, = *x.shape, x.device
    
    