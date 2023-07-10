# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        # dict(type='TensorboardLoggerHook')
        # dict(type='PaviLoggerHook') # for internal services
    ])
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
# load_from = '/home/sidd_s/scratch/saved_models/DDP/ddp_convnext_l_4x4_512x1024_160k_cityscapes.pth' ## usually when experimenting with our hypothesis 
load_from = None ## when using author original code 
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
