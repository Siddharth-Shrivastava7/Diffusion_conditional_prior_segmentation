'''
Self/Structure/Semantic Similarity based Diffusion model
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

from ..discrete_diffusion.utils import custom_schedule, similarity_transition_mat, logits_to_categorical
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
                transition_mat_type = 'matrix_expo', 
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
        self.beta_schedule_custom = beta_schedule_custom 
        self.beta_schedule_custom_start = beta_schedule_custom_start  
        self.beta_schedule_custom_end = beta_schedule_custom_end
        self.confusion = confusion
        self.k_nn = k_nn
        self.transition_mat_type = transition_mat_type
        self.bt = custom_schedule(self.beta_schedule_custom_start, self.beta_schedule_custom_end, self.timesteps, type=self.beta_schedule_custom)
        
        
        ## base one step transition matrices #  Construct transition matrices for q(x_t|x_{t-1}) 
        self.q_onestep_mats =  [
            similarity_transition_mat(self.bt, t, self.confusion_matrix, self.transition_mat_type, self.confusion, self.k_nn, matrix_expo_cumulative = False) \
            for t  in range(0, self.timesteps)
        ]
        assert self.q_onestep_mats.shape == (self.timesteps,
                                         self.num_classes + 1,
                                         self.num_classes + 1) 
        
        ## base cumulative transition matrices  # Construct transition matrices for q(x_t|x_start) 
        if self.transition_mat_type == 'matrix_expo':    
            self.q_mats = [
                        similarity_transition_mat(self.bt, t, self.confusion_matrix, self.transition_mat_type, self.confusion, self.k_nn, matrix_expo_cumulative = True) \
                        for t  in range(0, self.timesteps)
            ]
            self.q_mats = torch.stack(self.q_mats, dim=0)
        else: 
            q_mat_t = self.q_onestep_mats[0]
            self.q_mats = [q_mat_t]
            for t in range(1, self.timesteps):
                q_mat_t = torch.tensordot(q_mat_t, self.q_onestep_mats[t],
                                        dims=[[1], [0]])
                self.q_mats.append(q_mat_t)
            self.q_mats = torch.stack(self.q_mats, dim=0)  
        assert self.q_mats.shape == (self.timesteps,
                                         self.num_classes + 1,
                                         self.num_classes + 1) 
        
        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        self.transpose_q_onestep_mats = torch.permute(self.q_onestep_mats,
                                                  dims=(0, 2, 1))    
        
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
        '''
            t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
            itself.
        '''
        # sample time ## discrete time sample 
        times = torch.randint(0, self.timesteps, (batch, ), device=self.device).long()  
        ## corrupt the gt in its discrete space 
        ## self.q_pred has to change 
        noised_gt = self.q_pred(gt_down, times, 
                                   self.num_classes + 1, self.q_mats) ## noised_gt has a categorical label entries
        
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
        x_t = torch.randint(0, self. num_classes+1, [b,h,w], device=self.device).long() # stationary distribution 
        outs = list()
        for i in reversed(range(0, self.timesteps)): ## reverse traversing the diffusion pipeline 
            times = (torch.ones((b,), device=self.device) * i).long()
            ## p_logits would come in here 
            
            
            
            
            if self.accumulation: ## accumulating all the logits of x0_pred
                outs.append(mask_start_pred_logit.softmax(1))
        if self.accumulation: ## accumulating all the logits of x0_pred
            mask_start_pred_logit = torch.cat(outs, dim=0)
        logit = mask_start_pred_logit.mean(dim=0, keepdim=True)     
        return logit    
    
    
    def q_probs(self, q_mats, t, x_var_t):
        '''
            when q_mats is cumulative then, calculating  probabilities of q(x_t | x_0)
            
            when q_mats is onestep mats then, calculating  probabilities of q(x_t | x_{t-1}) 
        '''
        B, H, W = x_var_t.shape  
        q_mats_t = torch.index_select(q_mats, dim=0, index=t)
        x_var_t_onehot = F.one_hot(x_var_t.view(B, -1).to(torch.int64), self.num_classes + 1).to(torch.float64)
        out = torch.matmul(x_var_t_onehot, q_mats_t)  
        out = out.view(B, self.num_classes + 1, H, W)  ## probabilities of q(x_t | x_0)
        return out 
    
    def q_sample(self, q_probs): 
        '''
            Sampling from either q(x_t | x_0) or q(x_t | x_{t-1}) (i.e. add noise to the data). 
            sampling from categorical distribution is done via gumbel max trick!
        '''
        logits = torch.log(q_probs + torch.finfo(torch.float32).eps)  # eps approx 1e-7
        out_sample = logits_to_categorical(logits)
        return out_sample 
    
    def q_posterior_logits(self, x_start_pred_logits, x_t, t):
        """ Compute logits of q(x_{t-1} | x_t, x_start)."""
        
        fact1 = self.q_probs(self.transpose_q_onestep_mats, t, x_t) # x_{t} x Q_t^{T}
        fact2 = self.q_probs(self.q_mats, t - 1, F.Softmax(x_start_pred_logits, dim=1)) # x_{0} x Q_{t-1}^{\hat}
        tzero_logits = x_start_pred_logits
        
        # At t=0 we need the logits of q(x_{-1}|x_0, x_start)
        # where x_{-1} == x_start. This should be equal the log of x_0.
        out = torch.log(fact1 + torch.finfo(torch.float32).eps) + torch.log(fact2 + torch.finfo(torch.float32).eps) # log(fact1*fact2) = log(fact1) + log(fact2) 
        t_broadcast = torch.broadcast_to(t[0], x_start_pred_logits.shape)
        return torch.where(
            t_broadcast == 0, 
            tzero_logits, 
            out
        )
        
    def p_logits(self, img_feats, img_metas, x_t, t): 
        """
            x0_parameterisation 
            
            Predict the logits of p(x_{t-1}|x_t) by parameterizing this distribution
            as = sum_{pred_x_start} q(x_{t-1}, x_t |pred_x_start)p(pred_x_start|x_t) 
            = q(x_t |x_{t-1})q(x_{t-1}|pred_x_start), where pred_x_start ~ p(pred_x_start|x_t) ; approximating the expectation over p(pred_x_start|x_t) via single sample 
            
            can also refer: "https://beckham.nz/2022/07/11/d3pms.html"
        """
        input_times = self.time_mlp(t)
        ## converting the discrete labels into embedding (or one-hot, later if req) for passing into denoising network 
        x_t_emb = self.embedding_table(x_t).squeeze(1).permute(0, 3, 1, 2)
        # conditional input 
        feat = torch.cat([img_feats, x_t_emb], dim=1)
        feat = self.transform(feat)
        # denoising the mast at current time t
        x_start_pred_logits = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  
        # x_start_pred = torch.argmax(x_start_pred_logit, dim=1)
        
        t_broadcast = torch.broadcast_to(t[0], x_start_pred_logits.shape)
        model_logits = torch.where(
            t_broadcast == 0, 
            x_start_pred_logits, 
            self.q_posterior_logits(x_start_pred_logits, x_t, t)
        )
        
        assert (model_logits.shape == x_start_pred_logits.shape == \
            (x_t.shape[0], self.num_classes+1) + tuple(x_t.shape[2:]))
        
        return model_logits, x_start_pred_logits

    def p_sample(self, img_feats, img_metas, x_t, t): 
        
        model_logits, x_start_pred_logits = self.p_logits(img_feats, img_metas, x_t, t)
        
        # No noise when t == 0
        # NOTE: for t=0 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case. 
        
        nonzero_mask = torch.broadcast_to((t[0]!=0)*1 , x_start_pred_logits.shape)
        
        uniform_noise = torch.rand_like(model_logits)
        ## # To avoid numerical issues clip the uniform noise to a minimum value
        uniform_noise = torch.clamp(uniform_noise, min=torch.finfo(uniform_noise.dtype).tiny, max=1.)
        gumbel_noise = - torch.log(-torch.log(uniform_noise))
        sample = (model_logits + nonzero_mask * gumbel_noise).argmax(dim=1)
        
        return sample, F.Softmax(x_start_pred_logits, dim=1)