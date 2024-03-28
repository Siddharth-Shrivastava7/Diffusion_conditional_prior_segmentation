# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class FoggyZurichDataset(CityscapesDataset):
    """FoggyZurichDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            **kwargs)
