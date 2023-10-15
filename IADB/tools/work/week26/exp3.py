## writing down all the config files here only for simplicity in running the train file   

'''
    making it ditto to the one we have used to train the vainf deeplabv3+ model
'''  

norm_cfg = dict(type='BN', requires_grad=True) 
model = dict(
    type='EncoderDecoder',
    pretrained='mmcls://mobilenet_v2',
    backbone=dict(
        type='MobileNetV2',
        widen_factor=1.0,
        strides=(1, 2, 2, 1, 1, 1, 1),
        dilations=(1, 1, 1, 2, 2, 4, 4),
        out_indices=(1, 2, 4, 6)),
    decode_head=dict(
        type='DepthwiseSeparableASPPHead',
        in_channels=320,
        in_index=3,
        channels=512,
        dilations=(1, 12, 24, 36),
        c1_in_channels=24,
        c1_channels=48,
        dropout_ratio=0.1,
        num_classes=19, 
        # out_channels = 1,  ## binary seg 
        # threshold = 0.5, ## binary seg 
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict( 
            type='CrossEntropy2d', loss_weight=1.0, ignore_label=255, ignore_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18])),  
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=96,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19, 
        # out_channels = 1,  ## binary seg 
        # threshold = 0.5, ## binary seg 
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropy2d', loss_weight=0.4, ignore_label=255, ignore_classes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18])), 
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

crop_size = (512, 1024) # wanting to verify vainf code so not doing (512, 1024) 

train_pipeline = [
dict(type='LoadImageFromFile', to_float32=True),
dict(type='LoadAnnotations'),
dict(type='Resize', img_scale=(1024, 1024), keep_ratio = False),
dict(type='MyTransform_binary'),
dict(type='DefaultFormatBundle'),
dict(type='Collect', keys=['img', 'gt_semantic_seg'], meta_keys = ('filename', 'ori_filename', 'ori_shape', 'img_shape'))
]  

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True), # since not using normalisation so have to use float conversion
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1080),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='MyValTransform_binary'),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=4, # batch size since using only one gpu
    workers_per_gpu=4,
    train=dict(
        type='DarkZurichDataset' ,
        data_root='/home/sidd_s/scratch/dataset/dannet_perturb_acdc_gt/' ,
        img_dir='images/train',
        ann_dir='labels/train',
        pipeline=train_pipeline),
    val=dict(
        type='DarkZurichDataset' ,
        data_root='/home/sidd_s/scratch/dataset/dannet_perturb_acdc_gt/' ,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline),
    test=dict(
        type='DarkZurichDataset' ,
        data_root='/home/sidd_s/scratch/dataset/dannet_perturb_acdc_gt/' ,
        img_dir='images/val',
        ann_dir='labels/val',
        pipeline=test_pipeline)  
        )   

load_from = None # cause training with only 1 class 
resume_from = None
# Set up working dir to save files and logs. ## have to change this when changing experiments!
work_dir = '/home/sidd_s/scratch/mmseg_work_dir/week26/exp3/'  

cudnn_benchmark = True
seed = 0 
device='cuda' 
gpu_ids = range(1) # 4 gpus will be used 

log_config = dict(  
    interval=1, hooks=[dict(type='TextLoggerHook', by_epoch=True)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
workflow = [('train', 1), ('val', 1)] 
cudnn_benchmark = True
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
lr_config = None
runner = dict(type='EpochBasedRunner', max_epochs=500)
checkpoint_config = dict(by_epoch=True, interval=100)
evaluation = dict(interval=1, metric='mIoU', save_best='mIoU') 
