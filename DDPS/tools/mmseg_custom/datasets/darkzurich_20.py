# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS
from .cityscapes_20 import Cityscapes20Dataset


@DATASETS.register_module()
class DarkZurich20Dataset(Cityscapes20Dataset):
    """DarkZurichDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_rgb_anon.png',
            seg_map_suffix='_gt_labelTrainIds.png',
            **kwargs)
