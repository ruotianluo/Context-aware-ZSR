# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""VGG16 from https://arxiv.org/abs/1409.1556."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils

from modeling.ResNet import freeze_params

def vgg_detectron_weight_mapping(model):
    mapping_to_detectron = {}
    for k in model.state_dict():
        if '.weight' in k:
            mapping_to_detectron.update({k: k.replace('.weight', '_w')})
        if '.bias' in k:
            mapping_to_detectron.update({k: k.replace('.bias', '_b')})
    orphan_in_detectron = []

    return mapping_to_detectron, orphan_in_detectron


class VGG16_conv5_body(nn.Module):
    def __init__(self):
        super().__init__()
        cfg = [[64, 64, 'M'], [128, 128, 'M'], [256, 256, 256, 'M'], [512, 512, 512, 'M'], [512, 512, 512]] # Prune the conv5 max pool
        dim_in = 3
        for i in range(len(cfg)):
            for j in range(len(cfg[i])):
                if cfg[i][j] == 'M':
                    setattr(self, 'pool%d'%(i+1), nn.MaxPool2d(kernel_size=2, stride=2))
                else:
                    setattr(self, 'conv%d_%d'%(i+1,j+1), nn.Conv2d(dim_in, cfg[i][j], kernel_size=3, padding=1))
                    setattr(self, 'relu%d_%d'%(i+1,j+1), nn.ReLU(inplace=True))
                    dim_in = cfg[i][j]
        self.spatial_scale = 1. / 16.
        self.dim_out = dim_in

        self._init_modules()
    
    def _init_modules(self):
        for i, m in enumerate(self.children()):
            # Fix to conv3
            if i < 10:
                freeze_params(m)

    def detectron_weight_mapping(self):
        return vgg_detectron_weight_mapping(self)

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


class VGG16_roi_fc_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        self.fc6 = nn.Linear(dim_in * 7 * 7, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.dim_out = 4096

    def detectron_weight_mapping(self):
        return vgg_detectron_weight_mapping(self)

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=7,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = F.relu(self.fc6(x.view(x.size(0), -1)), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)
        return x