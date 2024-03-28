# Copyright (c) OpenMMLab. All rights reserved.
from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class CityscapesDarkBrightDataset(CityscapesDataset):
    """CityscapesDarkBrightDataset dataset."""

    def __init__(self, **kwargs):
        super().__init__(
            **kwargs)
