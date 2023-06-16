# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityscapesFoggyDataset(CityscapesDataset):
    """CityscapesFoggyDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_leftImg8bit_foggy_beta_0.005.png', ## may need to change later, as there many options available!
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)
