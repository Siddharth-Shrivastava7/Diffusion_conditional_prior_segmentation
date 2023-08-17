'''
Self/Structure Similarity based Diffusion model
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

from ..discrete_diffusion.schedule_mod import q_mats_from_onestepsdot, q_pred, custom_schedule
from ..discrete_diffusion.confusion_matrix import calculate_confusion_matrix_segformerb2

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, [0, 1])  # padding the last dimension
        return emb

@SEGMENTORS.register_module()
class SSD(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """
    def __init__(self,
                timesteps = 20, 
                band_diagonal = False,
                matrix_expo = True,
                confusion = True,
                k_nn = 3,
                beta_schedule_custom = 'expo', 
                beta_schedule_custom_start = -5.5, 
                beta_schedule_custom_end = -4.5,
                **kwargs):
        super(SSD, self).__init__(**kwargs)
        
        self.timesteps = timesteps 
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.num_classes + 1) # instead of one hot encoding making class embedding module for discrete data space ## (20,20)
        self.transform = ConvModule(
            self.decode_head.in_channels[0] + (self.num_classes + 1),
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ) # used for converting concatenated encoded i/p image and encoded;corrupted gt map to feature maps of dimension being half of the joint dimension of the concatenated inputs
        
        self.confusion_matrix = calculate_confusion_matrix_segformerb2()
        self.band_diagonal = band_diagonal
        self.matrix_expo = matrix_expo
        self.beta_schedule_custom = beta_schedule_custom 
        self.beta_schedule_custom_start = beta_schedule_custom_start  
        self.beta_schedule_custom_end = beta_schedule_custom_end
        self.confusion = confusion
        self.k_nn = k_nn
        
        self.bt = custom_schedule(self.beta_schedule_custom_start, self.beta_schedule_custom_end, self.timesteps, type=self.beta_schedule_custom)
        
        ## base transition matrices     
        self.q_mats = q_mats_from_onestepsdot(self.bt, self.timesteps, self.confusion_matrix, self.band_diagonal, self.matrix_expo, self.confusion, self.k_nn) 
        
        # time embeddings   
        time_dim = self.decode_head.in_channels[0] * 4  # 1024  ## like DDP 
        sinu_pos_emb = SinusoidalPosEmb(dim=self.decode_head.in_channels[0]) ## here dim could be 16 (as well, if going similar to DDP) ## but D3PM, multinomial diffusion and CCDM, also DDPS are taking input dim as #channels and out dim as  4*#ch...so this way of input channels being 256 procedding
        fourier_dim = self.decode_head.in_channels[0] ## same dimension in discrete space 
        ## similar to DDP 
        self.time_mlp = nn.Sequential(  # [2,] # is the input shape 
            sinu_pos_emb,  # [2, 256] # output shape
            nn.Linear(fourier_dim, time_dim),  # [256, 1024] # output shape
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [1024, 1024] # output shape
        )
        
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
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes
        
        ## corruption of discrete data gt 
        # sample time ## discrete time sample 
        times = torch.randint(0, self.timesteps, (batch, ), device=self.device).long() 
        ## corrupt the gt in its discrete space 
        noised_gt = q_pred(gt_down, times, 
                                   self.num_classes, self.q_mats)
        noised_gt_emb = self.embedding_table(noised_gt).squeeze(1).permute(0, 3, 1, 2) # encoding of gt when passing down the denoising net ## later may also need to try with one-hot encoding 
        
        ## conditional input 
        feat = torch.cat([x, noised_gt_emb], dim = 1)
        feat = self.transform(feat) 
        
        losses = dict()
        input_times = self.time_mlp(times)
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                [x], img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses
    
    def _decode_head_forward_train(self, x, t, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, t, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, t, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, t, img_metas, self.test_cfg)
        return seg_logits
    
    def encode_decode(self, img, img_metas): ## it is being called at the test time! 
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)[0] # encoding the image {both backbone and neck{fpn + multistagemerging}}
        if self.diffusion == "uniform":
            pass 
        elif self.diffusion == 'similarity':
            out = self.similarity_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    
    @torch.no_grad() 
    def similarity_sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        mask_t = torch.randint(0, self. num_classes+1, [b,h,w], device=self.device).long() # stationary distribution 
        outs = list()
        for i in reversed(range(0, self.timesteps)): ## reverse traversing the diffusion pipeline 
            times = (torch.ones((b,), device=self.device) * i).long()
            input_times = self.time_mlp(times)
            ## converting the discrete labels into embedding (or one-hot, later if req) for passing into denoising network 
            mask_t_emb = self.embedding_table(mask_t).squeeze(1).permute(0, 3, 1, 2)
            # conditional input 
            feat = torch.cat([x, mask_t_emb], dim=1)
            feat = self.transform(feat)
            # denoising the mast at current time t
            mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  
            mask_pred = torch.argmax(mask_logit, dim=1) ## predicted mask_x0 from time t, now using this have to calc mask @ t-1 time through posterior calc 
            ## posterior q(x_t-1 | x_t, x_0) 
            
            
            if self.accumulation:
                outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)     
        return logit    
            

        