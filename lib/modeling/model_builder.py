from functools import wraps
import importlib
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.config import cfg
from model.roi_layers import ROIPool, ROIAlign
import modeling.rpn_heads as rpn_heads
import modeling.rel_heads as rel_heads
import modeling.fast_rcnn_heads as fast_rcnn_heads
import modeling.mask_rcnn_heads as mask_rcnn_heads
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads
import utils.blob as blob_utils
import utils.net as net_utils
import utils.pretrained_weights_helper as pretrained_utils

logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name. func_name must identify a
    function in this module or the path to a function relative to the base
    'modeling' module.
    """
    if func_name == '':
        return None
    try:
        parts = func_name.split('.')
        # Refers to a function in this module
        if len(parts) == 1:
            return globals()[parts[0]]
        # Otherwise, assume we're referencing a module under modeling
        module_name = 'modeling.' + '.'.join(parts[:-1])
        module = importlib.import_module(module_name)
        return getattr(module, parts[-1])
    except Exception:
        logger.error('Failed to find function: %s', func_name)
        raise


def compare_state_dict(sa, sb):
    if sa.keys() != sb.keys():
        return False
    for k, va in sa.items():
        if not torch.equal(va, sb[k]):
            return False
    return True


def check_inference(net_func):
    @wraps(net_func)
    def wrapper(self, *args, **kwargs):
        if not self.training:
            if cfg.PYTORCH_VERSION_LESS_THAN_040:
                return net_func(self, *args, **kwargs)
            else:
                with torch.no_grad():
                    return net_func(self, *args, **kwargs)
        else:
            raise ValueError('You should call this function only on inference.'
                              'Set the network in inference mode by net.eval().')

    return wrapper


class Generalized_RCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # For cache
        self.mapping_to_detectron = None
        self.orphans_in_detectron = None

        # Backbone for feature extraction
        self.Conv_Body = get_func(cfg.MODEL.CONV_BODY)()

        # Region Proposal Network
        if cfg.RPN.RPN_ON and not cfg.MODEL.TAGGING:
            self.RPN = rpn_heads.generic_rpn_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)
        elif cfg.MODEL.TAGGING:
            self.RPN = rpn_heads.single_scale_gt_outputs(
                self.Conv_Body.dim_out, self.Conv_Body.spatial_scale)

        if cfg.FPN.FPN_ON:
            # Only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # RPN max level can be >= to ROI max level
            assert cfg.FPN.RPN_MAX_LEVEL >= cfg.FPN.ROI_MAX_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max/coarsest level to min/finest level)
            self.num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1

            # Retain only the spatial scales that will be used for RoI heads. `Conv_Body.spatial_scale`
            # may include extra scales that are used for RPN proposals, but not for RoI heads.
            self.Conv_Body.spatial_scale = self.Conv_Body.spatial_scale[-self.num_roi_levels:]

        # BBOX Branch
        if not cfg.MODEL.RPN_ONLY:
            self.Box_Head = get_func(cfg.FAST_RCNN.ROI_BOX_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            self.Box_Outs = fast_rcnn_heads.fast_rcnn_outputs(
                self.Box_Head.dim_out)

        if cfg.MODEL.NUM_RELATIONS > 0:
            self.Rel_Outs = rel_heads.rel_outputs(self.Box_Head.dim_out)
            if cfg.TEST.USE_REL_INFER:
                self.Rel_Inf = rel_heads.rel_infer(self.Rel_Outs)
            if cfg.REL_INFER.TRAIN:
                self.Rel_Inf_Train = rel_heads.rel_infer_train(self.Rel_Outs)

        # Mask Branch
        if cfg.MODEL.MASK_ON:
            self.Mask_Head = get_func(cfg.MRCNN.ROI_MASK_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Mask_Head, 'SHARE_RES5', False):
                self.Mask_Head.share_res5_module(self.Box_Head.res5)
            self.Mask_Outs = mask_rcnn_heads.mask_rcnn_outputs(self.Mask_Head.dim_out)

        # Keypoints Branch
        if cfg.MODEL.KEYPOINTS_ON:
            self.Keypoint_Head = get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD)(
                self.RPN.dim_out, self.roi_feature_transform, self.Conv_Body.spatial_scale)
            if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                self.Keypoint_Head.share_res5_module(self.Box_Head.res5)
            self.Keypoint_Outs = keypoint_rcnn_heads.keypoint_outputs(self.Keypoint_Head.dim_out)

        self._init_modules()

    def _init_modules(self):
        if cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
            if 'MobileNet' in cfg.MODEL.CONV_BODY:
                pretrained_utils.mobilenet_load_pretrained_imagenet_weights(self)
            elif 'VGG16' in cfg.MODEL.CONV_BODY:
                pretrained_utils.vgg16_load_pretrained_imagenet_weights(self)
            elif 'VGGM' in cfg.MODEL.CONV_BODY:
                pretrained_utils.vggm_load_pretrained_imagenet_weights(self)
            else:
                pretrained_utils.load_pretrained_imagenet_weights(self)
            # Check if shared weights are equaled
            if cfg.MODEL.MASK_ON and getattr(self.Mask_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Mask_Head.res5.state_dict(), self.Box_Head.res5.state_dict())
            if cfg.MODEL.KEYPOINTS_ON and getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                assert compare_state_dict(self.Keypoint_Head.res5.state_dict(), self.Box_Head.res5.state_dict())

        if cfg.TRAIN.FREEZE_CONV_BODY:
            for p in self.Conv_Body.parameters():
                p.requires_grad = False

    def forward(self, data, im_info, roidb=None, rois=None, **rpn_kwargs):
        if cfg.PYTORCH_VERSION_LESS_THAN_040:
            return self._forward(data, im_info, roidb, rois, **rpn_kwargs)
        else:
            with torch.set_grad_enabled(self.training):
                return self._forward(data, im_info, roidb, rois, **rpn_kwargs)

    def _forward(self, data, im_info, roidb=None, rois=None, **rpn_kwargs):
        im_data = data
        if self.training:
            roidb = list(map(lambda x: blob_utils.deserialize(x)[0], roidb))

        device_id = im_data.get_device()

        return_dict = {}  # A dict to collect return variables

        if cfg.MODEL.RELATION_NET_INPUT == 'GEO' and not cfg.REL_INFER.TRAIN and cfg.MODEL.NUM_RELATIONS > 0 and self.training:
            blob_conv = im_data
        else:
            blob_conv = self.Conv_Body(im_data)

        rpn_ret = self.RPN(blob_conv, im_info, roidb)
        # If rpn_ret doesn't have rpn_ret, the rois should have been given, we use that.
        if 'rois' not in rpn_ret:
            rpn_ret['rois'] = rois
        if hasattr(self, '_ignore_classes') and 'labels_int32' in rpn_ret:
            # Turn ignore classes labels to 0, because they are treated as background
            rpn_ret['labels_int32'][np.isin(rpn_ret['labels_int32'], self._ignore_classes)] = 0

        # if self.training:
        #     # can be used to infer fg/bg ratio
        #     return_dict['rois_label'] = rpn_ret['labels_int32']

        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]

        if not self.training:
            return_dict['blob_conv'] = blob_conv

        if not cfg.MODEL.RPN_ONLY:
            if cfg.MODEL.SHARE_RES5 and self.training:
                box_feat, res5_feat = self.Box_Head(blob_conv, rpn_ret)
            else:
                if cfg.MODEL.RELATION_NET_INPUT == 'GEO' and not cfg.REL_INFER.TRAIN and cfg.MODEL.NUM_RELATIONS > 0 and self.training:
                    box_feat = blob_conv.new_zeros(rpn_ret['labels_int32'].shape[0], self.Box_Head.dim_out)
                else:
                    box_feat = self.Box_Head(blob_conv, rpn_ret)
            if cfg.MODEL.RELATION_NET_INPUT == 'GEO' and not cfg.REL_INFER.TRAIN and cfg.MODEL.NUM_RELATIONS > 0 and self.training:
                cls_score = box_feat.new_zeros(rpn_ret['labels_int32'].shape[0], cfg.MODEL.NUM_CLASSES)
                bbox_pred = box_feat.new_zeros(rpn_ret['labels_int32'].shape[0], 4*cfg.MODEL.NUM_CLASSES)
            else:
                cls_score, bbox_pred = self.Box_Outs(box_feat)
            if cfg.TEST.TAGGING:
                rois_label = torch.from_numpy(rpn_ret['labels_int32'].astype('int64')).to(cls_score.device)
                accuracy_cls = cls_score.max(1)[1].eq(rois_label).float().mean(dim=0)
                print('Before refine:', accuracy_cls.item())

        if cfg.MODEL.NUM_RELATIONS > 0 and self.training:
            rel_scores, rel_labels = self.Rel_Outs(rpn_ret['rois'], box_feat, rpn_ret['labels_int32'], roidb=roidb)
        elif cfg.MODEL.NUM_RELATIONS > 0 and cfg.TEST.USE_REL_INFER:
            if cfg.TEST.TAGGING:
                rel_scores = self.Rel_Outs(rpn_ret['rois'], box_feat)
                cls_score = self.Rel_Inf(cls_score, rel_scores, roidb=roidb)
            else: # zero shot detection
                filt = (cls_score.topk(cfg.TEST.REL_INFER_PROPOSAL, 1)[1] == 0).float().sum(1) == 0
                filt[cls_score[:,1:].max(1)[0].topk(min(100, cls_score.shape[0]), 0)[1]] += 1
                filt = filt >= 2
                if filt.sum() == 0:
                    print('all background?')
                else:
                    tmp_rois = rpn_ret['rois'][filt.cpu().numpy().astype('bool')]
                    
                    rel_scores = self.Rel_Outs(tmp_rois, box_feat[filt])
                    tmp_cls_score = self.Rel_Inf(cls_score[filt], rel_scores, roidb=None)
                    cls_score[filt] = tmp_cls_score
            if cfg.TEST.TAGGING:
                rois_label = torch.from_numpy(rpn_ret['labels_int32'].astype('int64')).to(cls_score.device)
                accuracy_cls = cls_score.max(1)[1].eq(rois_label).float().mean(dim=0)
                print('After refine:', accuracy_cls.item())

        if self.training:
            return_dict['losses'] = {}
            return_dict['metrics'] = {}
            # rpn loss
            if not cfg.MODEL.TAGGING:
                rpn_kwargs.update(dict(
                    (k, rpn_ret[k]) for k in rpn_ret.keys()
                    if (k.startswith('rpn_cls_logits') or k.startswith('rpn_bbox_pred'))
                ))
                loss_rpn_cls, loss_rpn_bbox = rpn_heads.generic_rpn_losses(**rpn_kwargs)
                if cfg.FPN.FPN_ON:
                    for i, lvl in enumerate(range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1)):
                        return_dict['losses']['loss_rpn_cls_fpn%d' % lvl] = loss_rpn_cls[i]
                        return_dict['losses']['loss_rpn_bbox_fpn%d' % lvl] = loss_rpn_bbox[i]
                else:
                    return_dict['losses']['loss_rpn_cls'] = loss_rpn_cls
                    return_dict['losses']['loss_rpn_bbox'] = loss_rpn_bbox

            if not cfg.MODEL.RPN_ONLY:
                # bbox loss
                loss_cls, loss_bbox, accuracy_cls = fast_rcnn_heads.fast_rcnn_losses(
                    cls_score, bbox_pred, rpn_ret['labels_int32'], rpn_ret['bbox_targets'],
                    rpn_ret['bbox_inside_weights'], rpn_ret['bbox_outside_weights'])
                return_dict['losses']['loss_cls'] = loss_cls
                return_dict['losses']['loss_bbox'] = loss_bbox
                return_dict['metrics']['accuracy_cls'] = accuracy_cls

                if hasattr(loss_cls, 'mean_similarity'):
                    return_dict['metrics']['mean_similarity'] = loss_cls.mean_similarity

                if cfg.FAST_RCNN.SAE_REGU:
                    tmp = cls_score.sae_loss[(rpn_ret['labels_int32']>0).tolist()]
                    return_dict['losses']['loss_sae'] = 1e-4 * tmp.mean() if tmp.numel() > 0 else tmp.new_tensor(0)
                    if torch.isnan(return_dict['losses']['loss_sae']):
                        import pdb;pdb.set_trace()

            if cfg.REL_INFER.TRAIN:
                return_dict['losses']['loss_rel_infer'] = cfg.REL_INFER.TRAIN_WEIGHT * \
                    self.Rel_Inf_Train(rpn_ret['rois'], rpn_ret['labels_int32'], cls_score, rel_scores, roidb=roidb)

            if cfg.MODEL.MASK_ON:
                if getattr(self.Mask_Head, 'SHARE_RES5', False):
                    mask_feat = self.Mask_Head(res5_feat, rpn_ret,
                                               roi_has_mask_int32=rpn_ret['roi_has_mask_int32'])
                else:
                    mask_feat = self.Mask_Head(blob_conv, rpn_ret)
                mask_pred = self.Mask_Outs(mask_feat)
                # return_dict['mask_pred'] = mask_pred
                # mask loss
                loss_mask = mask_rcnn_heads.mask_rcnn_losses(mask_pred, rpn_ret['masks_int32'])
                return_dict['losses']['loss_mask'] = loss_mask

            if cfg.MODEL.KEYPOINTS_ON:
                if getattr(self.Keypoint_Head, 'SHARE_RES5', False):
                    # No corresponding keypoint head implemented yet (Neither in Detectron)
                    # Also, rpn need to generate the label 'roi_has_keypoints_int32'
                    kps_feat = self.Keypoint_Head(res5_feat, rpn_ret,
                                                  roi_has_keypoints_int32=rpn_ret['roi_has_keypoint_int32'])
                else:
                    kps_feat = self.Keypoint_Head(blob_conv, rpn_ret)
                kps_pred = self.Keypoint_Outs(kps_feat)
                # return_dict['keypoints_pred'] = kps_pred
                # keypoints loss
                if cfg.KRCNN.NORMALIZE_BY_VISIBLE_KEYPOINTS:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'])
                else:
                    loss_keypoints = keypoint_rcnn_heads.keypoint_losses(
                        kps_pred, rpn_ret['keypoint_locations_int32'], rpn_ret['keypoint_weights'],
                        rpn_ret['keypoint_loss_normalizer'])
                return_dict['losses']['loss_kps'] = loss_keypoints

            # pytorch0.4 bug on gathering scalar(0-dim) tensors
            for k, v in return_dict['losses'].items():
                return_dict['losses'][k] = v.unsqueeze(0)
            for k, v in return_dict['metrics'].items():
                return_dict['metrics'][k] = v.unsqueeze(0)

        else:
            # Testing
            return_dict.update(rpn_ret)
            if not cfg.MODEL.RPN_ONLY:
                if cfg.TEST.KEEP_HIGHEST:
                    cls_score = F.softmax(cls_score * 1e10, dim=1) * cls_score
                return_dict['cls_score'] = cls_score
                return_dict['bbox_pred'] = bbox_pred

        return return_dict

    def roi_feature_transform(self, blobs_in, rpn_ret, blob_rois='rois', method='RoIPoolF',
                              resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. The sampling_ratio argument
        is supported for some, but not all, RoI transform methods.

        RoIFeatureTransform abstracts away:
          - Use of FPN or not
          - Specifics of the transform method
        """
        assert method in {'RoIPoolF', 'RoIAlign'}, \
            'Unknown pooling method: {}'.format(method)

        if isinstance(blobs_in, list):
            # FPN case: add RoIFeatureTransform to each FPN level
            device_id = blobs_in[0].get_device()
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                if len(rpn_ret[bl_rois]):
                    rois = Variable(torch.from_numpy(rpn_ret[bl_rois])).cuda(device_id)
                    if method == 'RoIPoolF':
                        # Warning!: Not check if implementation matches Detectron
                        xform_out = ROIPool((resolution, resolution), sc)(bl_in, rois)
                    elif method == 'RoIAlign':
                        xform_out = ROIAlign(
                            (resolution, resolution), sc, sampling_ratio)(bl_in, rois)
                    bl_out_list.append(xform_out)

            # The pooled features from all levels are concatenated along the
            # batch dimension into a single 4D tensor.
            xform_shuffled = torch.cat(bl_out_list, dim=0)

            # Unshuffle to match rois from dataloader
            device_id = xform_shuffled.get_device()
            restore_bl = rpn_ret[blob_rois + '_idx_restore_int32']
            restore_bl = Variable(
                torch.from_numpy(restore_bl.astype('int64', copy=False))).cuda(device_id)
            xform_out = xform_shuffled[restore_bl]
        else:
            # Single feature level
            # rois: holds R regions of interest, each is a 5-tuple
            # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
            # rectangle (x1, y1, x2, y2)
            device_id = blobs_in.get_device()
            rois = Variable(torch.from_numpy(rpn_ret[blob_rois])).cuda(device_id)
            if method == 'RoIPoolF':
                xform_out = ROIPool((resolution, resolution), spatial_scale)(blobs_in, rois)
            elif method == 'RoIAlign':
                xform_out = ROIAlign(
                    (resolution, resolution), spatial_scale, sampling_ratio)(blobs_in, rois)

        return xform_out

    @check_inference
    def convbody_net(self, data):
        """For inference. Run Conv Body only"""
        blob_conv = self.Conv_Body(data)
        if cfg.FPN.FPN_ON:
            # Retain only the blobs that will be used for RoI heads. `blob_conv` may include
            # extra blobs that are used for RPN proposals, but not for RoI heads.
            blob_conv = blob_conv[-self.num_roi_levels:]
        return blob_conv

    @check_inference
    def mask_net(self, blob_conv, rpn_blob):
        """For inference"""
        mask_feat = self.Mask_Head(blob_conv, rpn_blob)
        mask_pred = self.Mask_Outs(mask_feat)
        return mask_pred

    @check_inference
    def keypoint_net(self, blob_conv, rpn_blob):
        """For inference"""
        kps_feat = self.Keypoint_Head(blob_conv, rpn_blob)
        kps_pred = self.Keypoint_Outs(kps_feat)
        return kps_pred

    @property
    def detectron_weight_mapping(self):
        if self.mapping_to_detectron is None:
            d_wmap = {}  # detectron_weight_mapping
            d_orphan = []  # detectron orphan weight list
            for name, m_child in self.named_children():
                if list(m_child.parameters()):  # if module has any parameter
                    child_map, child_orphan = m_child.detectron_weight_mapping()
                    d_orphan.extend(child_orphan)
                    for key, value in child_map.items():
                        new_key = name + '.' + key
                        d_wmap[new_key] = value
            self.mapping_to_detectron = d_wmap
            self.orphans_in_detectron = d_orphan

        return self.mapping_to_detectron, self.orphans_in_detectron

    def _add_loss(self, return_dict, key, value):
        """Add loss tensor to returned dictionary"""
        return_dict['losses'][key] = value
