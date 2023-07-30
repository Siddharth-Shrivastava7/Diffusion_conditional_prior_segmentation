norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint = '/home/sidd_s/scratch/saved_models/DDPS/segformer_b2_multistep_cityscapes/best_mIoU_iter_128000.pth'
model = dict(
    type='EncoderDecoderDiffusion',
    freeze_parameters=['backbone', 'decode_head'],
    pretrained=
    '/home/sidd_s/scratch/saved_models/DDPS/segformer_b2_multistep_cityscapes/best_mIoU_iter_128000.pth',
    backbone=dict(
        type='MixVisionTransformerCustomInitWeights',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[3, 4, 6, 3],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    decode_head=dict(
        type='SegformerHeadUnetFCHeadMultiStep',
        pretrained=
        '/home/sidd_s/scratch/saved_models/DDPS/segformer_b2_multistep_cityscapes/best_mIoU_iter_128000.pth',
        dim=256,
        out_dim=256,
        unet_channels=272,
        dim_mults=[1, 1, 1],
        cat_embedding_dim=16,
        diffusion_timesteps=20,
        collect_timesteps=[
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
            19
        ],
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=20,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        ignore_index=0,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
dataset_type = 'DarkZurich20Dataset'
data_root = '/home/sidd_s/scratch/dataset/DarkZurich'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        # img_scale=(1024, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    test=dict(
        type='DarkZurich20Dataset',
        data_root='/home/sidd_s/scratch/dataset/DarkZurich',
        img_dir='rgb_anon/val',
        ann_dir='gt/val',
        pipeline=test_pipeline))

log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
optimizer = dict(
    type='AdamW', lr=0.00015, betas=[0.9, 0.96], weight_decay=0.045)
optimizer_config = dict()
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-06,
    step=20000,
    gamma=0.5,
    min_lr=1e-06,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=16000, max_keep_ckpts=1)
evaluation = dict(
    interval=16000, metric='mIoU', pre_eval=True, save_best='mIoU')
custom_hooks = [
    dict(
        type='ConstantMomentumEMAHook',
        momentum=0.01,
        interval=25,
        eval_interval=16000,
        auto_resume=True,
        priority=49)
]
work_dir = './work_dirs/segformer_mit_b2_segformer_head_unet_fc_small_multi_step_ade_pretrained_freeze_embed_160k_DarkZurich20_finetune_ema'
gpu_ids = range(0, 8)
auto_resume = True
