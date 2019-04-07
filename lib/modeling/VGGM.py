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
from modeling.VGG16 import vgg_detectron_weight_mapping

class SpatialCrossMapLRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, k=1, ACROSS_CHANNELS=True):
        super(SpatialCrossMapLRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(self.k).pow(self.beta)
        x = x.div(div)
        return x

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class VGGM_conv5_body(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,96,(7, 7),(2, 2))
        self.relu1 = nn.ReLU(True)
        self.norm1 = SpatialCrossMapLRN(5, 0.0005, 0.75, 2)
        self.pool1 = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        self.conv2 = nn.Conv2d(96,256,(5, 5),(2, 2),(1, 1))
        self.relu2 = nn.ReLU(True)
        self.norm2 = SpatialCrossMapLRN(5, 0.0005, 0.75, 2)
        self.pool2 = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        self.conv3 = nn.Conv2d(256,512,(3, 3),(1, 1),(1, 1))
        self.relu3 = nn.ReLU(True)
        self.conv4 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu4 = nn.ReLU(True)
        self.conv5 = nn.Conv2d(512,512,(3, 3),(1, 1),(1, 1))
        self.relu5 = nn.ReLU(True)
        self.spatial_scale = 1. / 16.
        self.dim_out = 512

        self._init_modules()
    
    def _init_modules(self):
        # Fix conv1
        freeze_params(self.conv1)

    def detectron_weight_mapping(self):
        return vgg_detectron_weight_mapping(self)

    def forward(self, x):
        for m in self.children():
            x = m(x)
        return x


class VGGM_roi_fc_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        self.fc6 = nn.Linear(dim_in * 6 * 6, 4096)
        self.fc7 = nn.Linear(4096, 4096)

        self.dim_out = 4096

    def detectron_weight_mapping(self):
        return vgg_detectron_weight_mapping(self)

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=6,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        x = F.relu(self.fc6(x.view(x.size(0), -1)), inplace=True)
        x = F.relu(self.fc7(x), inplace=True)
        return x