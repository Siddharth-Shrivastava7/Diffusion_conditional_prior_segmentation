# dataset settings
dataset_type = 'ACDCRainDataset'
data_root = '/home/sidd_s/scratch/dataset/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',   
        img_scale=(1920, 1080),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False, ## for MS: multiscale testing:: flip is True and img_ratios are uncommented they are inherently taken accont in the code while using the "--aug-test" arg.
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='acdc_trainval/rgb_anon/rain/val', 
        ann_dir = 'acdc_gt/gt/rain/val',
        pipeline=test_pipeline))
