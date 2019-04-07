import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.config import cfg
import nn as mynn
import utils.net as net_utils

from modeling.ResNet import freeze_params

from collections import namedtuple, OrderedDict, Iterable

# ---------------------------------------------------------------------------- #
# Generic MobileNet components
# ---------------------------------------------------------------------------- #

class MobileNet_v1_conv12_body(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv, self.dim_out = mobilenet_base(V1_CONV_DEFS[:12])
        self.conv = nn.Sequential(*self.conv)

        self.spatial_scale = 1 / 16

        self._init_modules()

    def _init_modules(self):
        assert 0 <= cfg.MOBILENET.FREEZE_AT <= 12
        for i in range(cfg.RESNETS.FREEZE_AT):
            freeze_params(self.conv[i])

        # Freeze all bn (affine) layers !!!
        self.apply(freeze_bn)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        for i in range(cfg.MOBILENET.FREEZE_AT):
            self.conv[i].eval()
        
        self.apply(freeze_bn)

    def forward(self, x):
        return self.conv(x)

class MobileNet_v2_conv14_body(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv, self.dim_out = mobilenet_base(V2_CONV_DEFS[:6])
        self.conv = nn.Sequential(*self.conv)

        self.spatial_scale = 1 / 16

        self._init_modules()

    def _init_modules(self):
        assert 0 <= cfg.MOBILENET.FREEZE_AT <= 14
        for i in range(cfg.RESNETS.FREEZE_AT):
            freeze_params(self.conv[i])

        # Freeze all bn (affine) layers !!!
        self.apply(freeze_bn)

    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)

        for i in range(cfg.MOBILENET.FREEZE_AT):
            self.conv[i].eval()
        
        self.apply(freeze_bn)

    def forward(self, x):
        return self.conv(x)


class MobileNet_roi_conv_head(nn.Module):
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        self.stride_init = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION // 7
        self.avgpool = nn.AvgPool2d(7)

    def _init_modules(self):
        # Freeze all bn (affine) layers !!!
        self.apply(freeze_bn)
    
    def train(self, mode=True):
        # Override train so that the training mode is set as we want
        nn.Module.train(self, mode)
        self.apply(freeze_bn)

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        feat = self.conv(x)
        x = self.avgpool(feat)
        if cfg.MODEL.SHARE_RES5 and self.training:
            return x, feat 
        else:
            return x

class MobileNet_v1_roi_conv_head(MobileNet_roi_conv_head):

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__(dim_in, roi_xform_func, spatial_scale)

        tmp_conv_def = V1_CONV_DEFS[12:]
        tmp_conv_def[0] = tmp_conv_def[0]._replace(stride=self.stride_init)

        self.conv, self.dim_out = mobilenet_base(tmp_conv_def, in_channels = dim_in)
        self.conv = nn.Sequential(OrderedDict(zip([str(_) for _ in range(12, 12+len(self.conv))], self.conv)))

        self._init_modules()


class MobileNet_v2_roi_conv_head(MobileNet_roi_conv_head):

    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__(dim_in, roi_xform_func, spatial_scale)

        tmp_conv_def = V2_CONV_DEFS[6:]
        tmp_conv_def[0] = tmp_conv_def[0]._replace(stride=self.stride_init)
        self.conv, self.dim_out = mobilenet_base(tmp_conv_def, in_channels = dim_in)
        self.conv = nn.Sequential(OrderedDict(zip([str(_) for _ in range(14, 14+len(self.conv))], self.conv)))

        self._init_modules()

# Set batchnorm always in eval mode during training
def freeze_bn(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()
        freeze_params(m)


# ---------------------------------------------------------------------------- #
# Get MobileNet
# ---------------------------------------------------------------------------- #

class Conv2d_tf(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_tf, self).__init__(*args, **kwargs)
        self.padding = kwargs.get('padding', 'SAME')
        kwargs['padding'] = 0
        if not isinstance(self.stride, Iterable):
            self.stride = (self.stride, self.stride)
        if not isinstance(self.dilation, Iterable):
            self.dilation = (self.dilation, self.dilation)

    def forward(self, input):
        # from https://github.com/pytorch/pytorch/issues/3867
        if self.padding == 'VALID':
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            padding=0,
                            dilation=self.dilation, groups=self.groups)
        input_rows = input.size(2)
        filter_rows = self.weight.size(2)
        effective_filter_size_rows = (filter_rows - 1) * self.dilation[0] + 1
        out_rows = (input_rows + self.stride[0] - 1) // self.stride[0]
        padding_rows = max(0, (out_rows - 1) * self.stride[0] + effective_filter_size_rows -
                                input_rows)
        # padding_rows = max(0, (out_rows - 1) * self.stride[0] +
        #                         (filter_rows - 1) * self.dilation[0] + 1 - input_rows)
        rows_odd = (padding_rows % 2 != 0)
        # same for padding_cols
        input_cols = input.size(3)
        filter_cols = self.weight.size(3)
        effective_filter_size_cols = (filter_cols - 1) * self.dilation[1] + 1
        out_cols = (input_cols + self.stride[1] - 1) // self.stride[1]
        padding_cols = max(0, (out_cols - 1) * self.stride[1] + effective_filter_size_cols -
                                input_cols)
        # padding_cols = max(0, (out_cols - 1) * self.stride[1] +
        #                         (filter_cols - 1) * self.dilation[1] + 1 - input_cols)
        cols_odd = (padding_cols % 2 != 0)

        if rows_odd or cols_odd:
            input = F.pad(input, [0, int(cols_odd), 0, int(rows_odd)])

        return F.conv2d(input, self.weight, self.bias, self.stride,
                        padding=(padding_rows // 2, padding_cols // 2),
                        dilation=self.dilation, groups=self.groups)

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def depth_multiplier_v2(depth,
                        multiplier,
                        divisible_by=8,
                        min_depth=8):
    d = depth
    return _make_divisible(d * multiplier, divisible_by,
                                                    min_depth)

Conv = namedtuple('Conv', ['kernel', 'stride', 'depth'])
DepthSepConv = namedtuple('DepthSepConv', ['stride', 'depth'])
InvertedResidual = namedtuple('InvertedResidual', ['stride', 'depth', 'num', 't']) # t is the expension factor

V1_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    DepthSepConv(stride=1, depth=64),
    DepthSepConv(stride=2, depth=128),
    DepthSepConv(stride=1, depth=128),
    DepthSepConv(stride=2, depth=256),
    DepthSepConv(stride=1, depth=256),
    DepthSepConv(stride=2, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=1, depth=512),
    DepthSepConv(stride=2, depth=1024),
    DepthSepConv(stride=1, depth=1024)
]

V2_CONV_DEFS = [
    Conv(kernel=3, stride=2, depth=32),
    InvertedResidual(stride=1, depth=16, num=1, t=1),
    InvertedResidual(stride=2, depth=24, num=2, t=6),
    InvertedResidual(stride=2, depth=32, num=3, t=6),
    InvertedResidual(stride=2, depth=64, num=4, t=6),
    InvertedResidual(stride=1, depth=96, num=3, t=6),
    InvertedResidual(stride=2, depth=160, num=3, t=6),
    InvertedResidual(stride=1, depth=320, num=1, t=6),
    Conv(kernel=1, stride=1, depth=1280),
]

Conv2d = Conv2d_tf
# Conv2d = nn.Conv2d

class _conv_bn(nn.Module):
    def __init__(self, inp, oup, kernel, stride):
        super(_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            Conv2d(inp, oup, kernel, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _conv_dw(nn.Module):
    def __init__(self, inp, oup, stride):
        super(_conv_dw, self).__init__()
        self.conv = nn.Sequential(
            # dw
            nn.Sequential(
            Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True)
            ),
            # pw
            nn.Sequential(
            Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True)
            )
        )
        self.depth = oup

    def forward(self, x):
        return self.conv(x)


class _inverted_residual_bottleneck(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(_inverted_residual_bottleneck, self).__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Sequential(
            Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True)
            ) if expand_ratio > 1 else nn.Sequential(),
            # dw
            nn.Sequential(
            Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True)
            ),
            # pw-linear
            nn.Sequential(
            Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup)
            )
        )
        self.depth = oup
        
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


def mobilenet_base(conv_defs=V1_CONV_DEFS, depth=lambda x: x, in_channels=3):
    layers = []
    for conv_def in conv_defs:
        if isinstance(conv_def, Conv):
            layers += [_conv_bn(in_channels, depth(conv_def.depth), conv_def.kernel, conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, DepthSepConv):
            layers += [_conv_dw(in_channels, depth(conv_def.depth), conv_def.stride)]
            in_channels = depth(conv_def.depth)
        elif isinstance(conv_def, InvertedResidual):
          for n in range(conv_def.num):
            stride = conv_def.stride if n == 0 else 1
            layers += [_inverted_residual_bottleneck(in_channels, depth(conv_def.depth), stride, conv_def.t)]
            in_channels = depth(conv_def.depth)
    return layers, in_channels

class MobileNet(nn.Module):

    def __init__(self, version='1', depth_multiplier=1.0, min_depth=8, num_classes=1001, dropout=0.2):
        super(MobileNet, self).__init__()
        self.dropout = dropout
        conv_defs = V1_CONV_DEFS if version == '1' else V2_CONV_DEFS
        
        if version == '1':
            depth = lambda d: max(int(d * depth_multiplier), min_depth)
            self.features, out_channels = mobilenet_base(conv_defs=conv_defs, depth = depth)
        else:
            # Change the last layer of self.features
            depth = lambda d: depth_multiplier_v2(d, depth_multiplier, min_depth=min_depth)
            self.features, out_channels = mobilenet_base(conv_defs=conv_defs[:-1], depth=depth)
            depth = lambda d: depth_multiplier_v2(d, max(depth_multiplier, 1.0), min_depth=min_depth)
            tmp, out_channels = mobilenet_base(conv_defs=conv_defs[-1:], in_channels=out_channels, depth=depth)
            self.features = self.features + tmp

        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Conv2d(out_channels, num_classes, 1)
        
        for m in self.modules():
            if 'BatchNorm' in m.__class__.__name__:
                m.eps = 0.001
                m.momentum = 0.003

    def forward(self, x):
        x = self.features(x)
        x = x.mean(2, keepdim=True).mean(3, keepdim=True)
        x = F.dropout(x, self.dropout, self.training)
        x = self.classifier(x)
        x = x.squeeze(3).squeeze(2)
        return x
