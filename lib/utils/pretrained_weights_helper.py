"""
Helper functions for converting resnet pretrained weights from other formats
"""
import os
import pickle

import torch

import nn as mynn
import utils.detectron_weight_helper as dwh
from core.config import cfg


def load_pretrained_imagenet_weights(model):
    """Load pretrained weights
    Args:
        num_layers: 50 for res50 and so on.
        model: the generalized rcnnn module
    """
    _, ext = os.path.splitext(cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
    if ext == '.pkl':
        with open(cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS, 'rb') as fp:
            src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrianed_state_dict = src_blobs
    else:
        weights_file = os.path.join(cfg.ROOT_DIR, cfg.RESNETS.IMAGENET_PRETRAINED_WEIGHTS)
        pretrianed_state_dict = convert_state_dict(torch.load(weights_file))

        # Convert batchnorm weights
        for name, mod in model.named_modules():
            if isinstance(mod, mynn.AffineChannel2d):
                if cfg.FPN.FPN_ON:
                    pretrianed_name = name.split('.', 2)[-1]
                else:
                    pretrianed_name = name.split('.', 1)[-1]
                bn_mean = pretrianed_state_dict[pretrianed_name + '.running_mean']
                bn_var = pretrianed_state_dict[pretrianed_name + '.running_var']
                scale = pretrianed_state_dict[pretrianed_name + '.weight']
                bias = pretrianed_state_dict[pretrianed_name + '.bias']
                std = torch.sqrt(bn_var + 1e-5)
                new_scale = scale / std
                new_bias = bias - bn_mean * scale / std
                pretrianed_state_dict[pretrianed_name + '.weight'] = new_scale
                pretrianed_state_dict[pretrianed_name + '.bias'] = new_bias

    model_state_dict = model.state_dict()

    pattern = dwh.resnet_weights_name_pattern()

    name_mapping, _ = model.detectron_weight_mapping

    for k, v in name_mapping.items():
        if isinstance(v, str):  # maybe a str, None or True
            if pattern.match(v):
                if cfg.FPN.FPN_ON:
                    pretrianed_key = k.split('.', 2)[-1]
                else:
                    pretrianed_key = k.split('.', 1)[-1]
                if ext == '.pkl':
                    model_state_dict[k].copy_(torch.Tensor(pretrianed_state_dict[v]))
                else:
                    model_state_dict[k].copy_(pretrianed_state_dict[pretrianed_key])


def convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    for k, v in src_dict.items():
        toks = k.split('.')
        if k.startswith('layer'):
            assert len(toks[0]) == 6
            res_id = int(toks[0][5]) + 1
            name = '.'.join(['res%d' % res_id] + toks[1:])
            dst_dict[name] = v
        elif k.startswith('fc'):
            continue
        else:
            name = '.'.join(['res1'] + toks)
            dst_dict[name] = v
    return dst_dict


def mobilenet_load_pretrained_imagenet_weights(model):
    """Load pretrained weights
    Args:
        model: the generalized rcnnn module
    """
    _, ext = os.path.splitext(cfg.TRAIN.IMAGENET_PRETRAINED_WEIGHTS)
    if ext == '.pkl':
        with open(cfg.TRAIN.IMAGENET_PRETRAINED_WEIGHTS, 'rb') as fp:
            src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrianed_state_dict = src_blobs
    else:
        weights_file = os.path.join(cfg.ROOT_DIR, cfg.TRAIN.IMAGENET_PRETRAINED_WEIGHTS)
        pretrianed_state_dict = mobilenet_convert_state_dict(torch.load(weights_file))
    model.Conv_Body.conv.load_state_dict(pretrianed_state_dict, strict=False)
    if hasattr(model, 'Box_Head'):
        model.Box_Head.conv.load_state_dict(pretrianed_state_dict, strict=False)

def mobilenet_convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    dst_dict = {}
    for k, v in src_dict.items():
        if 'features' in k:
            dst_dict[k[len('features.'):]] = v
    return dst_dict

def vgg_load_pretrained_imagenet_weights(model, convert_state_dict):
    """Load pretrained weights
    Args:
        model: the generalized rcnnn module
    """
    _, ext = os.path.splitext(cfg.TRAIN.IMAGENET_PRETRAINED_WEIGHTS)
    if ext == '.pkl':
        with open(cfg.TRAIN.IMAGENET_PRETRAINED_WEIGHTS, 'rb') as fp:
            src_blobs = pickle.load(fp, encoding='latin1')
        if 'blobs' in src_blobs:
            src_blobs = src_blobs['blobs']
        pretrianed_state_dict = src_blobs
    else:
        weights_file = os.path.join(cfg.ROOT_DIR, cfg.TRAIN.IMAGENET_PRETRAINED_WEIGHTS)
        pretrianed_state_dict = convert_state_dict(torch.load(weights_file))
    model.Conv_Body.load_state_dict(pretrianed_state_dict, strict=False)
    if hasattr(model, 'Box_Head'):
        model.Box_Head.load_state_dict(pretrianed_state_dict, strict=False)

def vgg16_load_pretrained_imagenet_weights(model):
    vgg_load_pretrained_imagenet_weights(model, vgg16_convert_state_dict)

def vgg16_convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    mapping = {'features.0': 'conv1_1',
            'features.2': 'conv1_2',
            'features.5': 'conv2_1',
            'features.7': 'conv2_2',
            'features.10': 'conv3_1',
            'features.12': 'conv3_2',
            'features.14': 'conv3_3',
            'features.17': 'conv4_1',
            'features.19': 'conv4_2',
            'features.21': 'conv4_3',
            'features.24': 'conv5_1',
            'features.26': 'conv5_2',
            'features.28': 'conv5_3',
            'classifier.0': 'fc6',
            'classifier.3': 'fc7',
            'classifier.6': 'fc8'}

    dst_dict = {}

    for k in mapping.keys():
        dst_dict[mapping[k]+'.weight'] = src_dict[k+'.weight']
        dst_dict[mapping[k]+'.bias'] = src_dict[k+'.bias']

    return dst_dict


def vggm_load_pretrained_imagenet_weights(model):
    vgg_load_pretrained_imagenet_weights(model, vggm_convert_state_dict)

def vggm_convert_state_dict(src_dict):
    """Return the correct mapping of tensor name and value

    Mapping from the names of torchvision model to our resnet conv_body and box_head.
    """
    mapping = {'features.0': 'conv1',
            'features.4': 'conv2',
            'features.8': 'conv3',
            'features.10': 'conv4',
            'features.12': 'conv5',
            'classifier.0': 'fc6',
            'classifier.3': 'fc7',
            'classifier.6': 'fc8'}

    dst_dict = {}

    for k in mapping.keys():
        dst_dict[mapping[k]+'.weight'] = src_dict[k+'.weight']
        dst_dict[mapping[k]+'.bias'] = src_dict[k+'.bias']

    return dst_dict