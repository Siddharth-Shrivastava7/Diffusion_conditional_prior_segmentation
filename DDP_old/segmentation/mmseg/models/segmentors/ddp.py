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





def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))


def alpha_cosine_log_snr(t, ns=0.0002, ds=0.00025):
    return -log((torch.cos((t + ns) / (1 + ds) * math.pi * 0.5) ** -2) - 1,
                eps=1e-5)  # not sure if this accounts for beta being clipped to 0.999 in discrete version


def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr)) # converting log(alpha) -> alpha, and log(alpha) -> (1 - alpha) 
''' 
    sort of 1st thing of above came from: 
        sqrt{1 / (1 + e^{log_snr})} = sqrt{1 / (1 + e^{-log(alpha)})} = sqrt{1 / (1 + (1/alpha))} = sqrt{alpha/(1 + alpha) ; now (1+alpha) -> 1 thus: 
        sqrt{alpha} 

    sort of 2nd thing of above came from: 
        rationalising (multiplying denominator and numerator) with (1 - aplha) so, its like: 
        (1-alpha) / {(1+aplha)*(1-aplha)} = (1-alpha) / (1-alpha^2) ; now, (1-alpha^2) -> 1 thus: 
        (1 - alpha) => sqrt{1-alpha}
'''


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


@SEGMENTORS.register_module()
class DDP(EncoderDecoder):
    """Encoder Decoder segmentors.
    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 bit_scale=0.1,
                 timesteps=1,
                 randsteps=1,
                 time_difference=1,
                 learned_sinusoidal_dim=16,
                 sample_range=(0, 0.999),
                 noise_schedule='cosine',
                 diffusion='ddim',
                 accumulation=False,
                 **kwargs):
        super(DDP, self).__init__(**kwargs)

        self.bit_scale = bit_scale
        self.timesteps = timesteps
        self.randsteps = randsteps
        self.diffusion = diffusion
        self.time_difference = time_difference
        self.sample_range = sample_range
        self.use_gt = False
        self.accumulation = accumulation
        self.embedding_table = nn.Embedding(self.num_classes + 1, self.decode_head.in_channels[0])

        print(f" timesteps: {timesteps},"
              f" randsteps: {randsteps},"
              f" sample_range: {sample_range},"
              f" diffusion: {diffusion}")

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.transform = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ) # used for converting concatenated encoded i/p image and encoded;corrupted gt map to feature maps of dimension being half of the joint dimension of the concatenated inputs

        # time embeddings
        time_dim = self.decode_head.in_channels[0] * 4  # 1024
        sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
        fourier_dim = learned_sinusoidal_dim + 1

        self.time_mlp = nn.Sequential(  # [2,] # is the input shape 
            sinu_pos_emb,  # [2, 17] # output shape
            nn.Linear(fourier_dim, time_dim),  # [17, 1024] # output shape
            nn.GELU(),
            nn.Linear(time_dim, time_dim)  # [1024, 1024] # output shape
        )
        
        
        ### new add-on-module
        self.transform_x_gtperturb = ConvModule(
            self.decode_head.in_channels[0] * 2,
            self.decode_head.in_channels[0],
            1,
            padding=0,
            conv_cfg=None,
            norm_cfg=None,
            act_cfg=None
        ) 

    def encode_decode(self, img, img_metas): ## it is being called at the test time! 
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)[0] # encoding the image {both backbone and neck{fpn + multistagemerging}}
        if self.diffusion == "ddim":
            out = self.ddim_sample(x, img_metas)
        elif self.diffusion == 'ddpm':
            out = self.ddpm_sample(x, img_metas)
        else:
            raise NotImplementedError
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def forward_train(self, img, img_metas, gt_semantic_seg): ## it is being called at the TRAIN time! 
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
        x = self.extract_feat(img)[0]  # bs, 256, h/4, w/4
        batch, c, h, w, device, = *x.shape, x.device
        gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        gt_down = gt_down.to(gt_semantic_seg.dtype)
        gt_down[gt_down == 255] = self.num_classes

        gt_down = self.embedding_table(gt_down).squeeze(1).permute(0, 3, 1, 2) # encoding of gt
        gt_down = (torch.sigmoid(gt_down) * 2 - 1) * self.bit_scale # encoding of gt 

        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                      self.sample_range[1])  # [bs]

        # random noise
        noise = torch.randn_like(gt_down)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_gt = alpha * gt_down + sigma * noise

        # conditional input
        feat = torch.cat([x, noised_gt], dim=1)
        feat = self.transform(feat) ## why reduce back to 256? (from 512) may be because decoder head was tuned to 256 channels input >> they used 6 layer deformable attention for that 

        losses = dict()
        input_times = self.time_mlp(noise_level)
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                [x], img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        return losses


    ## self aligned denoising training
    def forward_train_self_aligned_denoising(self, img, img_metas, gt_semantic_seg): ## it is being called at the TRAIN time!
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
        x = self.extract_feat(img)[0]  # bs, 256, h/4, w/4
        batch, c, h, w, device, = *x.shape, x.device
        
        ## extra addition :: concatinating perturbed gt feature with x :: can be according to curriculum learning, gradually increasing the pertuberation of gt, for now lets try with a small and fixed one 
        # gt_down = resize(gt_semantic_seg.float(), size=(h, w), mode="nearest")
        # gt_down = gt_down.to(gt_semantic_seg.dtype)
        # gt_down[gt_down == 255] = self.num_classes 
        # ## pertubation the gt 
        # any_random_class = torch.randint(0, 19, (1,))[0].item()
        # differnt_random_class = torch.randint(0, 19, (1,))[0].item()
        # while differnt_random_class==any_random_class: 
        #     differnt_random_class = torch.randint(0, 19, (1,))[0].item()  
        # gt_down_perturbed = gt_down.detach().clone()    
        # gt_down_perturbed[gt_down_perturbed == any_random_class] = 255
        # gt_down_perturbed[gt_down_perturbed == differnt_random_class] = any_random_class
        # gt_down_perturbed[gt_down_perturbed==255] = differnt_random_class
        # ## pertubation the gt 
        # gt_down_perturbed = self.embedding_table(gt_down_perturbed).squeeze(1).permute(0, 3, 1, 2) # encoding of gt
        # gt_down_perturbed = (torch.sigmoid(gt_down_perturbed) * 2 - 1) * self.bit_scale # encoding of gt 
        # # extra concat feat of original image encoding and perturbed gt # in order to improve it  
        # x = torch.cat([x, gt_down_perturbed], dim=1)
        # x = self.transform_x_gtperturb(x)
    
        # city_name = img_metas[0]['filename'].split('/')[-2] ## cityscapes prediction by Robustnet
        # pred_path = img_metas[0]['filename'].replace('dataset/cityscapes/leftImg8bit/val/' + city_name ,'results/oneformer/semantic_inference/') ## cityscapes prediction by Robustnet 
        # pred = torch.tensor(np.array(Image.open(pred_path))).to(device) # loading MIC prediction {as a starting point to correct it further}  
        # pred[pred==255] = self.num_classes ## for gt case 
        # pred = pred.view(1,1, pred.shape[0], pred.shape[1]) ## shape => (1, 1, 1080, 1920)
        # pred_down = resize(pred.float(), size=(h, w), mode="nearest")
        # ## passing to map_decoder! 
        
        ## orignial according self aligned training 
        ## map_t and concate feats    
        mask_t = torch.randn((batch, self.decode_head.in_channels[0], h, w), device=device) # instead of normal noise using gt perturbed noise  
        feat_init = torch.cat([x, mask_t], dim=1) # for decoding << before that concatenating the img encoding and corrupted gt map which is for sampling is the sample from the normal distribution >> 
        feat_init = self.transform(feat_init) ## converting (512 concat feats to 256 feats for having compatibility to decoder input module)
    
        ## map_pred 
        times_ones = torch.ones((batch,), device=device).float() 
        noise_level_ones = self.log_snr(times_ones)
        input_times_ones = self.time_mlp(noise_level_ones)
        map_pred_logits = self.decode_head.forward([feat_init], input_times_ones)
        map_pred = torch.argmax(map_pred_logits, dim=1)
        
        ##encode map_pred 
        map_enc = self.embedding_table(map_pred).squeeze(1).permute(0, 3, 1, 2) ## shape would be (b, 256, h/4, w/4)
        map_enc = (torch.sigmoid(map_enc) * 2 - 1) * self.bit_scale 
        
        # corrupt the map_enc
        # sample time
        times = torch.zeros((batch,), device=device).float().uniform_(self.sample_range[0],
                                                                      self.sample_range[1])  # [bs]
        # random noise
        noise = torch.randn_like(map_enc)
        noise_level = self.log_snr(times)
        padded_noise_level = self.right_pad_dims_to(img, noise_level)
        alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
        noised_map_enc = alpha * map_enc + sigma * noise
        ## concat noised_map_enc and feats
        feat = torch.cat([x, noised_map_enc], dim=1) 
        feat = self.transform(feat)
        # pred 
        losses = dict()
        input_times = self.time_mlp(noise_level)
        loss_decode = self._decode_head_forward_train([feat], input_times, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        
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

    def right_pad_dims_to(self, x, t):
        padding_dims = x.ndim - t.ndim
        if padding_dims <= 0:
            return t
        return t.view(*t.shape, *((1,) * padding_dims))

    def _get_sampling_timesteps(self, batch, *, device):
        times = []
        for step in range(self.timesteps):
            t_now = 1 - (step / self.timesteps) * (1 - self.sample_range[0])
            t_next = max(1 - (step + 1 + self.time_difference) / self.timesteps * (1 - self.sample_range[0]),
                         self.sample_range[0])
            time = torch.tensor([t_now, t_next], device=device)
            time = repeat(time, 't -> t b', b=batch)
            times.append(time)
        return times

    ## The modified one 
    # @torch.no_grad()
    # def ddim_sample(self, x, img_metas):
    #     b, c, h, w, device = *x.shape, x.device
    #     time_pairs = self._get_sampling_timesteps(b, device=device)
    #     x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
    #     ## below holds the modification code for inferecing ddp with the starting point as the prediction of either DA or DG model 
    #     # if img_metas[0]['filename'].find('dark_zurich')!=-1: ## dataset we are dealing with, requires DDP to act as a correction module
    #     if img_metas[0]['filename'].find('cityscapes')!=-1: ## dataset we are dealing with, requires DDP to act as a correction module
    #     # if img_metas[0]['filename'].find('bdd100k')!=-1:
    #         # ## loading the predicted image
    #         # pred_path = img_metas[0]['filename'].replace('rgb_anon/val/night/GOPR0356/','pred/segformer_pred/')
    #         # pred_path = img_metas[0]['filename'].replace('/rgb_anon/', '/gt/').replace('_rgb_anon.png','_gt_labelTrainIds.png') ## gt pred path for testing its upperlimit # Dz val testing 
    #         # pred_path = img_metas[0]['filename'].replace('/leftImg8bit/', '/gtFine/').replace('_leftImg8bit.png','_gtFine_labelTrainIds.png') ## gt path for cityscapes 
    #         # city_name = img_metas[0]['filename'].split('/')[-2] ## cityscapes prediction by Robustnet
    #         # pred_path = img_metas[0]['filename'].replace('dataset/cityscapes/leftImg8bit/val/' + city_name ,'results/oneformer/semantic_inference/') ## cityscapes prediction by Robustnet
    #         # pred_path = img_metas[0]['filename'].replace('rgb_anon/val_ref/day/GOPR0356_ref/','pred/mic_pred/') # when using day ref images for MIC pred DZ Day data input images
    #         # print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', img_metas[0]['filename']) # image path of the dataset 
    #         # pred_path = img_metas[0]['filename'].replace('/dataset/bdd100k_seg/bdd100k/seg/images/val/','/results/robustnet/bdd100k/saved_models/val/pred_trainids/').replace('.jpg', '.png') # when using day ref images for Robustnet pred BDD100k data input images
    #         # pred_path = img_metas[0]['filename'].replace('rgb_anon/val/night/GOPR0356/','pred/robustnet_pred/') # Robustnet prediction on DZ-val images
    #         city_name = img_metas[0]['filename'].split('/')[-2]  
    #         pred_path = img_metas[0]['filename'].replace('leftImg8bit/val/' + city_name, 'pred/segformerb2/')
    #         pred = torch.tensor(np.array(Image.open(pred_path))).to(device) # loading MIC prediction {as a starting point to correct it further}  
    #         pred[pred==255] = self.num_classes ## for gt case 
    #         pred = pred.view(1,1, pred.shape[0], pred.shape[1]) ## shape => (1, 1, 1080, 1920)
    #         ## have to resize pred::in order to bring it to shape of x ##(b,1,h/4, w/4) 
    #         pred_down = resize(pred.float(), size=(h, w), mode="nearest")
    #         # encoding the predicted image
    #         mask_enc = self.embedding_table(pred_down.long()).squeeze(1).permute(0, 3, 1, 2) ## shape would be (b, 256, h/4, w/4)
    #         mask_enc = (torch.sigmoid(mask_enc) * 2 - 1) * self.bit_scale 
    #         # print(mask_enc.shape) # torch.Size([1, 256, 256, 512])
    #         ## corrupting the predicted image 
    #         # sample time
    #         # times = torch.zeros((b,), device=device).float().uniform_(self.sample_range[0],
    #         #                                                             self.sample_range[1])  # [bs]
    #         # # print(times)
    #         # # random noise
    #         # noise = torch.randn_like(mask_enc)
    #         # noise_level = self.log_snr(times)
    #         # padded_noise_level = self.right_pad_dims_to(x, noise_level)
    #         # alpha, sigma = log_snr_to_alpha_sigma(padded_noise_level)
    #         # mask_t = alpha * mask_enc + sigma * noise           
    #         mask_t = mask_enc
    #     # mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device) # this is the "map_t" in the algorithm; which is the sample from the normal distribution # original
    #     outs = list()
    #     for idx, (times_now, times_next) in enumerate(time_pairs):
    #         # x_mod = torch.zeros_like(x)
    #         # feat = torch.cat([x_mod, mask_t], dim=1) 
    #         feat = torch.cat([x, mask_t], dim=1) # for decoding << before that concatenating the img encoding and corrupted gt map which is for sampling is the sample from the normal distribution >> 
    #         feat = self.transform(feat) ## converting (512 concat feats to 256 feats for having compatibility to decoder input module)
    #         # feat = mask_t # not working ...only features 
    #         log_snr = self.log_snr(times_now)
    #         log_snr_next = self.log_snr(times_next)

    #         padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
    #         padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
    #         alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
    #         alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

    #         input_times = self.time_mlp(log_snr)
    #         mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ] ## it is the map_pred :: the decoded y_0^{hat} from the map decoder 
    #         mask_pred = torch.argmax(mask_logit, dim=1) ## the label map from the map decoder logit output
    #         mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2) ## encoding the map pred from the map decoder part 
    #         mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale ## encoding the map pred from the map decoder part
    #         pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8) ## eps calculating exactly same as what was mentioned in the paper 
    #         mask_t = mask_pred * alpha_next + pred_noise * sigma_next ## this mask_t is basically the mask_{t+1}; they used it as mask_t since, it will be reused in the next iteration of the timesteps ## and this is main step of DDIM formulation...OLA!!!
    #         if self.accumulation:
    #             outs.append(mask_logit.softmax(1))
    #     if self.accumulation:
    #         mask_logit = torch.cat(outs, dim=0)
    #     logit = mask_logit.mean(dim=0, keepdim=True)
    #     return logit
    
    ## the original one by the author
    @torch.no_grad()
    def ddim_sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)
        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for idx, (times_now, times_next) in enumerate(time_pairs):
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)
            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale
            pred_noise = (mask_t - alpha * mask_pred) / sigma.clamp(min=1e-8)
            mask_t = mask_pred * alpha_next + pred_noise * sigma_next
            
            if self.accumulation:
                outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit

    @torch.no_grad()
    def ddpm_sample(self, x, img_metas):
        b, c, h, w, device = *x.shape, x.device
        time_pairs = self._get_sampling_timesteps(b, device=device)

        x = repeat(x, 'b c h w -> (r b) c h w', r=self.randsteps)
        mask_t = torch.randn((self.randsteps, self.decode_head.in_channels[0], h, w), device=device)
        outs = list()
        for times_now, times_next in time_pairs:
            feat = torch.cat([x, mask_t], dim=1)
            feat = self.transform(feat)

            log_snr = self.log_snr(times_now)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr = self.right_pad_dims_to(mask_t, log_snr)
            padded_log_snr_next = self.right_pad_dims_to(mask_t, log_snr_next)
            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            input_times = self.time_mlp(log_snr)
            mask_logit = self._decode_head_forward_test([feat], input_times, img_metas=img_metas)  # [bs, 150, ]
            mask_pred = torch.argmax(mask_logit, dim=1)
            mask_pred = self.embedding_table(mask_pred).permute(0, 3, 1, 2)
            mask_pred = (torch.sigmoid(mask_pred) * 2 - 1) * self.bit_scale

            c = -expm1(log_snr - log_snr_next)
            mean = alpha_next * (mask_t * (1 - c) / alpha + c * mask_pred)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)
            noise = torch.where(
                rearrange(times_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(mask_t),
                torch.zeros_like(mask_t)
            )
            mask_t = mean + (0.5 * log_variance).exp() * noise

            if self.accumulation:
                outs.append(mask_logit.softmax(1))
        if self.accumulation:
            mask_logit = torch.cat(outs, dim=0)
        logit = mask_logit.mean(dim=0, keepdim=True)
        return logit