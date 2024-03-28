# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityscapesRainyDataset(CityscapesDataset):
    """CityscapesRainyDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            img_suffix='_leftImg8bit_rain_alpha_0.02_beta_0.01_dropsize_0.005_pattern_12.png', ## may need to change later, as there many options available!
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)
