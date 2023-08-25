'''
Self/Structure/Semantic Similarity based Diffusion model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.core import add_prefix
from mmseg.ops import resize
from mmcv.cnn import ConvModule
import math
import numpy as np
import os
from PIL import Image 

from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder

from ..discrete_diffusion.utils import get_transition_rate, logits_to_categorical, get_powers, builder_fn
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
                schedule_steps = 20,  
                mutual_info_min_exponent = 1e-4, 
                mutual_info_max_exponent = 1e+5, 
                mutual_info_interpolation_steps = 256,  
                mutual_info_kind = 'linear', 
                allow_out_of_bounds=False,
                accumulation=False,
                **kwargs):
        super(SSD, self).__init__(**kwargs)
        
        self.schedule_steps = schedule_steps 
        self.embedding_table = nn.Embedding(self.num_classes+1, self.num_classes) ## instead of one-hot, playing with the embedding for discrete labels, and ignoring background feat in output of embedding            ### <<< can change later to one-hot, if req >>> ### 
        self.transform = ConvModule(
            self.decode_head.in_channels[0] + self.num_classes,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ) # used for converting concatenated encoded i/p image and encoded;corrupted gt map to feature maps of dimension being half of the joint dimension of the concatenated inputs
        self.accumulation = accumulation
        self.mutual_info_min_exponent = mutual_info_min_exponent 
        self.mutual_info_max_exponent = mutual_info_max_exponent  
        self.mutual_info_interpolation_steps = mutual_info_interpolation_steps 
        self.mutual_info_kind = mutual_info_kind 
        self.allow_out_of_bounds = allow_out_of_bounds
        
        ## forming transition rate matrix 
        self.confusion_matrix = calculate_confusion_matrix_segformerb2() ## confusion matrix is unnormalised here 
        ## get powers from Mutual information based noise schedule 
        self.transition_rate = get_transition_rate(self.confusion_matrix)  
        
        ## init cityscapes train data distribution 
        self.init_distribution_dict = np.load('/home/sidd_s/Diffusion_conditional_prior_segmentation/DDP/segmentation/mmseg/models/discrete_diffusion/confusion_similarity_results/cityscapes_gt_labels_init_distribution_without_background.npy',allow_pickle='TRUE').item()
        self.init_distribution = np.array(list(self.init_distribution_dict.values()))
        
        ## Finding exponent powers using mutual information based noise scheduling 
        self.powers = get_powers(
            self.schedule_steps, 
            self.transition_rate, 
            self.init_distribution,
            self.mutual_info_min_exponent, 
            self.mutual_info_max_exponent, 
            self.mutual_info_interpolation_steps, 
            self.mutual_info_kind, 
            self.allow_out_of_bounds
        )
            
        ## base one step transition matrices #  Construct transition matrices for q(x_t|x_{t-1}) 
        q_onestep_mats =  [
            torch.from_numpy(builder_fn(self.transition_rate, (self.powers[t+1] - self.powers[t]))) \
            for t  in range(0, self.schedule_steps)
        ]
        q_onestep_mats = torch.stack(q_onestep_mats, dim=0)
        ## background is added as an absorbing state in the forward diffusion (markov chain) process
        extended_q_onestep_mats = torch.zeros(q_onestep_mats.shape[0], 
                                              q_onestep_mats.shape[1] + 1,
                                              q_onestep_mats.shape[2] + 1) 
        extended_q_onestep_mats[:, :q_onestep_mats.shape[1], :q_onestep_mats.shape[2]] = q_onestep_mats
        extended_q_onestep_mats[:, -1, -1] = 1 ## background as the absorbing state 
        self.q_onestep_mats = extended_q_onestep_mats
        assert self.q_onestep_mats.shape == (self.schedule_steps,
                                         self.num_classes+1,
                                         self.num_classes+1)  
         
        ## base cumulative transition matrices  # Construct transition matrices for q(x_t|x_start) 
        q_mats = [
                    torch.from_numpy(builder_fn(self.transition_rate, self.powers[t+1])) \
                    for t  in range(0, self.schedule_steps)
        ]
        q_mats = torch.stack(q_mats, dim=0)
        ## background is added as an absorbing state in the forward diffusion (markov chain) process 
        extended_q_mats = torch.zeros(q_mats.shape[0], 
                                        q_mats.shape[1] + 1,
                                        q_mats.shape[2] + 1) 
        extended_q_mats[:, :q_mats.shape[1], :q_mats.shape[2]] = q_mats
        extended_q_mats[:, -1, -1] = 1 ## background as the absorbing state 
        self.q_mats = extended_q_mats
        assert self.q_mats.shape == (self.schedule_steps,
                                         self.num_classes+1,
                                         self.num_classes+1) 
        
        # Don't precompute transition matrices for q(x_{t-1} | x_t, x_start)
        # Can be computed from self.q_mats and self.q_one_step_mats.
        # Only need transpose of q_onestep_mats for posterior computation.
        self.transpose_q_onestep_mats = torch.permute(self.q_onestep_mats,
                                                  dims=(0, 2, 1))    
        
        
        # time embeddings   
        time_dim = self.decode_head.in_channels[0] * 4  # 1024  ## like DDP 
        sinu_pos_emb = SinusoidalPosEmb(dim=self.decode_head.in_channels[0]) ## here dim could be 16 (as well, if going similar to DDP) ## but D3PM, multinomial diffusion and CCDM, also DDPS are taking input dim as #channels and out dim as  4*#ch...so this way of input channels being 256 procedding  ### <<< can change later to 16, if req >>> ### 
        fourier_dim = self.decode_head.in_channels[0] ## same dimension in discrete space ### <<< can change later to 16, if req >>> ### 
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
        # backbone & neck
        img_feat = self.extract_feat(img)[0]  # bs, 256, h/4, w/4
        batch, c, h, w, device, = *img_feat.shape, img_feat.device
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest").long()
        gt_down[gt_down == 255] = self.num_classes # background, which will be an absorbing state in the markov chain
        gt_down = gt_down.view(batch, h, w) ## 'bhw'
        
        ## corruption of discrete data gt 
        '''
            t starts at zero. so x_0 is the first noisy datapoint, not the datapoint
            itself.
        '''
        # sample time ## discrete time sample 
        times = torch.randint(0, self.schedule_steps, (batch, ), device=device).long()  
        ## corrupt the gt in its discrete space 
        noised_gt = self.q_sample(self.q_probs(self.q_mats.to(device), times, gt_down)) 
        noised_gt_emb = self.embedding_table(noised_gt).squeeze(1).permute(0, 3, 1, 2) # encoding of gt when passing down the denoising net ## later may also need to try with one-hot encoding 
        
        ## conditional input 
        feat = torch.cat([img_feat, noised_gt_emb], dim = 1)
        feat = self.transform(feat) 
        
        losses = dict()
        input_times = self.time_mlp(times)
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                [img_feat], img_metas, gt_semantic_seg)
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
        img_feat = self.extract_feat(img)[0] # encoding the image {both backbone and neck{fpn + multistagemerging}}
        out_mask, out = self.similarity_sample(img_feat, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    
    @torch.no_grad() 
    def similarity_sample(self, img_feat, img_metas):
        b, c, h, w, device = *img_feat.shape, img_feat.device
        ## not needed to start from stationary, rather we need to start from prediction of the model we need to improve!
        # x = torch.randint(0, self.num_classes, [b,h,w], device=device).long() # stationary distribution  
        # < have to write x as the prediction output of the model (here segformerb2) :: have to make it for every other dataset; currently used for batch size of 1 >
        if img_metas[0]['filename'].find('cityscapes')!=-1: ## for cityscapes specifically 
            city_name = img_metas[0]['filename'].split('/')[-2]  
            pred_path = img_metas[0]['filename'].replace('leftImg8bit/val/' + city_name, 'pred/segformerb2/')
            x = torch.tensor(np.array(Image.open(pred_path))).to(device)
            x = x[None, None, :, :] ## expanding the dimension for batches and channels 
            x = resize(x.float(), size=(h, w), mode="nearest").long() 
            x[x==255] = self.num_classes ## if ever, background was present, then this! 
            x = x.view(1, h, w) # BHW 
            
        outs = list()
        for i in reversed(range(0, self.schedule_steps)): ## reverse traversing the diffusion pipeline 
            times = (torch.ones((b,), device=device) * i).long()
            x,  x_start_pred_logits = self.p_sample(img_feat, img_metas, 
                                                    x, times)
            if self.accumulation: ## accumulating all the probas of x0_pred
                outs.append(x_start_pred_logits.softmax(1))
        if self.accumulation: ## accumulating all the probas of x0_pred
            x_start_pred_logits = torch.cat(outs, dim=0)
        logit = x_start_pred_logits.mean(dim=0, keepdim=True)     
        return x, logit     
    
    
    def q_probs(self, q_mats, t, x_var_t, x_var_t_logits = False):
        '''
            when q_mats is cumulative then, calculating  probabilities of q(x_t | x_0)
            
            when q_mats is onestep mats then, calculating  probabilities of q(x_t | x_{t-1}) 
        '''
        if x_var_t_logits:
            B, C, H, W = x_var_t.shape  
            x_var_t_onehot_like = x_var_t.view(B, -1, C).to(torch.float64)
        else:
            B, H, W = x_var_t.shape  
            x_var_t_onehot_like = F.one_hot(x_var_t.view(B, -1).to(torch.int64), self.num_classes+1).to(torch.float64)
               
        q_mats_t = torch.index_select(q_mats.to(x_var_t.device), dim=0, index=t)
        out = torch.matmul(x_var_t_onehot_like, q_mats_t)  
        out = out.view(B, self.num_classes+1, H, W)  ## probabilities of q(x_t | x_0)
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
        
        fact1 = self.q_probs(self.transpose_q_onestep_mats.to(x_t.device), t, x_t) # x_{t} x Q_t^{T}
        fact2 = self.q_probs(self.q_mats.to(x_start_pred_logits.device), t - 1, F.softmax(x_start_pred_logits, dim=1), x_var_t_logits=True) # x_{0} x Q_{t-1}^{\hat}
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
        
    def p_logits(self, img_feat, img_metas, x_t, t): 
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
        feat = torch.cat([img_feat, x_t_emb], dim=1)
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
        
        assert (model_logits.shape == x_start_pred_logits.shape \
            == ((x_t.shape[0], self.num_classes+1) + tuple(x_t.shape[1:])))
        
        return model_logits, x_start_pred_logits

    def p_sample(self, img_feat, img_metas, x_t, t): 
        """Sample one timestep from the model p(x_{t-1} | x_t)."""
        
        model_logits, x_start_pred_logits = self.p_logits(img_feat, img_metas, x_t, t)
        
        # No noise when t == 0
        # NOTE: for t=0 this just "samples" from the argmax
        #   as opposed to "sampling" from the mean in the gaussian case. 
        
        nonzero_mask = torch.broadcast_to((t[0]!=0)*1 , x_start_pred_logits.shape)
        
        uniform_noise = torch.rand_like(model_logits)
        ## # To avoid numerical issues clip the uniform noise to a minimum value
        uniform_noise = torch.clamp(uniform_noise, min=torch.finfo(uniform_noise.dtype).tiny, max=1.)
        gumbel_noise = - torch.log(-torch.log(uniform_noise))
        sample = (model_logits + nonzero_mask * gumbel_noise).argmax(dim=1)
        
        assert sample.shape == x_t.shape
        assert x_start_pred_logits.shape == model_logits.shape
        
        return sample, x_start_pred_logits