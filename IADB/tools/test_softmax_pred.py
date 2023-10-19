# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import (load_checkpoint,
                         wrap_fp16_model)

from mmseg import digit_version
from mmseg.apis import  single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import  build_dp, get_device, setup_multi_processes


def main(config_path, checkpoint_path, gpu_id = 0):

    cfg = mmcv.Config.fromfile(config_path)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    cfg.gpu_ids = [gpu_id]
    distributed = False

    # build the val dataloader
    dataset_val  = build_dataset(cfg.data.test) 
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the val dataloader
    data_loader_val = build_dataloader(dataset_val, **test_loader_cfg)

    # # build the train dataloader < similar to test, just changing the directory > 
    # cfg.data.test 
    # dataset_train  = build_dataset(cfg.data.train)  
    # train_loader_cfg = {
    #     **loader_cfg,
    #     'samples_per_gpu': 1,
    #     'shuffle': False,  # Not shuffle by default
    #     **cfg.data.get('train_dataloader', {})
    # }
    # # build the train dataloader
    # data_loader_train = build_dataloader(dataset_train, **train_loader_cfg)
    # build the val dataloader

    train_test = cfg.data.test
    train_test.img_dir= 'leftImg8bit/train'
    train_test.ann_dir= 'gtFine/train'
    
    dataset_train_test  = build_dataset(train_test) 
    # The default loader config
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': 1,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the val dataloader
    data_loader_train_test = build_dataloader(dataset_train_test, **test_loader_cfg)


    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset_val.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset_val.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()

    cfg.device = get_device() 
    
    if not torch.cuda.is_available():
        assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
            'Please use MMCV >= 1.4.4 for CPU training!'
    model = revert_sync_batchnorm(model)
    model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids) 
    ## train results 
    train_results = single_gpu_test(
        model,  
        data_loader_train_test,
        softmaxop=True) ## hard coded for softmax predictions

    ## val results 
    val_results = single_gpu_test(
        model,  
        data_loader_val,
        softmaxop=True) ## hard coded for softmax predictions 

    return train_results,val_results  # results of softmax predictions of whole cityscapes dataset    
        

