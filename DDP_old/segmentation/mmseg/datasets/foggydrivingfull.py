# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class FoggyDrivingFullDataset(CityscapesDataset):
    """FoggyDrivingFull dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_labelTrainIds.png',
            **kwargs)
