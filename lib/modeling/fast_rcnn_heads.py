import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils

class word_embedding_linear(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        self.dim_in = dim_in
        if cfg.FAST_RCNN.LOSS_TYPE == 'cross_entropy':
            self.mlp = nn.Linear(300, dim_in + 1)
        else:
            self.mlp = nn.Linear(dim_in, 300, bias=not cfg.FAST_RCNN.SAE_REGU)
            if cfg.FAST_RCNN.SAE_REGU:
                torch.nn.init.orthogonal_(self.mlp.weight)
    
    def forward(self, x):
        if cfg.FAST_RCNN.LOSS_TYPE == 'cross_entropy':
            weight, bias = self.mlp(self.word_embedding).split(self.dim_in, 1)
            return torch.matmul(x, weight.t()) + bias.t()
        else:
            tmp = self.mlp(x)
            out =  torch.matmul(F.normalize(tmp), self.word_embedding.t())
            if cfg.FAST_RCNN.SAE_REGU:
                sae_loss = ((tmp.mm(self.mlp.weight) - x) ** 2).mean(1)
                out.sae_loss = sae_loss
            return out


class fast_rcnn_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        if not cfg.MODEL.WORD_EMBEDDING_REGU:
            self.cls_score = nn.Linear(dim_in, cfg.MODEL.NUM_CLASSES)
        else:
            self.cls_score = word_embedding_linear(dim_in)
        self.cls_score.register_buffer('word_embedding', torch.zeros(cfg.MODEL.NUM_CLASSES, 300))
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:  # bg and fg
            self.bbox_pred = nn.Linear(dim_in, 4 * 2)
        else:
            self.bbox_pred = nn.Linear(dim_in, 4 * cfg.MODEL.NUM_CLASSES)

        self._init_weights()

    def set_word_embedding(self, embedding):
        if embedding.shape[0] != cfg.MODEL.NUM_CLASSES - 1:
            print('WARNING: Word embedding size not matched')
            return
        self.cls_score.word_embedding[0].copy_(embedding.mean(0))
        self.cls_score.word_embedding[1:].copy_(embedding)
        self.cls_score.word_embedding.copy_(F.normalize(self.cls_score.word_embedding, dim=1))

    def _init_weights(self):
        if not cfg.MODEL.WORD_EMBEDDING_REGU:
            init.normal_(self.cls_score.weight, std=0.01)
            init.constant_(self.cls_score.bias, 0)
        init.normal_(self.bbox_pred.weight, std=0.001)
        init.constant_(self.bbox_pred.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'cls_score.weight': 'cls_score_w',
            'cls_score.bias': 'cls_score_b',
            'bbox_pred.weight': 'bbox_pred_w',
            'bbox_pred.bias': 'bbox_pred_b'
        }
        orphan_in_detectron = []
        return detectron_weight_mapping, orphan_in_detectron

    def forward(self, x):

        if x.dim() == 4:
            x = x.squeeze(3).squeeze(2)
        cls_score = _tmp_cls_score = self.cls_score(x)

        if hasattr(self, '_ignore_classes') or cfg.TEST.CONSE:
            # The score in shaded_classes will be -inf
            if cfg.TEST.CONSE:
                shaded_classes = cfg.TEST.CLASS_SPLIT['target']
            else:
                shaded_classes = self._ignore_classes
            # The ignored classes are treated as background
            tmp = torch.zeros_like(cls_score)
            tmp[:, 0] = cls_score[:, shaded_classes].sum(dim=1)
            tmp[:, shaded_classes] = float('-inf')
            cls_score = cls_score + tmp
        if cfg.TEST.CONSE or (cfg.MODEL.TAGGING and self.training) or (cfg.TEST.TAGGING and not self.training):
            tmp = cls_score.new_zeros(cfg.MODEL.NUM_CLASSES)
            tmp[0] = float('-inf')
            cls_score = cls_score + tmp
        if not self.training and cfg.FAST_RCNN.LOSS_TYPE == 'cross_entropy':
            cls_score = F.softmax(cls_score, dim=1)
            if cfg.TEST.CONSE:
                conse_feat = cls_score.mm(self.cls_score.word_embedding) # N x 300
                cls_score = F.normalize(conse_feat).mm(self.cls_score.word_embedding.t())
                if hasattr(self, '_ignore_classes'):
                    cls_score[:, self._ignore_classes] = -1
                if cfg.TEST.TAGGING: # Ignore the background embedding
                    cls_score[:, 0] = -1 # Since this will only happen in test time, so it's ok to inplace change values.

        bbox_pred = self.bbox_pred(x)
        if cfg.MODEL.TAGGING:
            bbox_pred = torch.zeros_like(bbox_pred)

        if cfg.FAST_RCNN.SAE_REGU:
            cls_score.sae_loss = _tmp_cls_score.sae_loss
        return cls_score, bbox_pred


def fast_rcnn_losses(cls_score, bbox_pred, label_int32, bbox_targets,
                     bbox_inside_weights, bbox_outside_weights):
    device_id = cls_score.get_device()
    rois_label = Variable(torch.from_numpy(label_int32.astype('int64'))).cuda(device_id)
    if cfg.FAST_RCNN.LOSS_TYPE in ['cross_entropy', 'triplet_softmax']:
        if cfg.FAST_RCNN.LOSS_TYPE == 'triplet_softmax':
            cls_score = cls_score * 3 # This method is borrowed from ji zhang's large scale relationship detection
        if not cfg.MODEL.TAGGING:
            loss_cls = F.cross_entropy(cls_score, rois_label)
        else:
            loss_cls = F.cross_entropy(cls_score, rois_label, ignore_index=0)
        if cfg.FAST_RCNN.LOSS_TYPE == 'triplet_softmax':
            cls_score = cls_score / 3
    else:
        if cfg.FAST_RCNN.LOSS_TYPE == 'multi_margin':
            loss_cls = F.multi_margin_loss(cls_score, rois_label,
                                            margin=cfg.FAST_RCNN.MARGIN,
                                            reduction='none')
        elif cfg.FAST_RCNN.LOSS_TYPE == 'max_margin':
            cls_score_with_high_target = cls_score.clone()
            cls_score_with_high_target.scatter_(1, rois_label.view(-1, 1), 1e10) # This make sure the following variable always has the target in the first column
            target_and_offender_index = cls_score_with_high_target.sort(1, True)[1][:, :2] # Target and the largest score excpet target
            loss_cls = F.multi_margin_loss(cls_score.gather(1, target_and_offender_index), rois_label.data * 0,
                                            margin=cfg.FAST_RCNN.MARGIN,
                                            reduction='none')
        loss_cls = loss_cls[rois_label > 0]
        loss_cls = loss_cls.mean() if loss_cls.numel() > 0 else loss_cls.new_tensor(0)
    
    # Secretly log the mean similarity!
    if cfg.FAST_RCNN.LOSS_TYPE in ['triplet_softmax', 'max_margin', 'multi_margin']:
        loss_cls.mean_similarity = cls_score[rois_label>0].gather(1, rois_label[rois_label>0].unsqueeze(1)).mean().detach() / 3

    bbox_targets = Variable(torch.from_numpy(bbox_targets)).cuda(device_id)
    bbox_inside_weights = Variable(torch.from_numpy(bbox_inside_weights)).cuda(device_id)
    bbox_outside_weights = Variable(torch.from_numpy(bbox_outside_weights)).cuda(device_id)
    loss_bbox = net_utils.smooth_l1_loss(
        bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
    if cfg.MODEL.TAGGING:
        loss_bbox = torch.zeros_like(loss_bbox)

    # class accuracy
    cls_preds = cls_score.max(dim=1)[1].type_as(rois_label)
    if not cfg.MODEL.TAGGING:
        accuracy_cls = cls_preds.eq(rois_label).float().mean(dim=0)
    else:
        accuracy_cls = cls_preds[rois_label > 0].eq(rois_label[rois_label > 0]).float().mean(dim=0) # Ignore index 0

    return loss_cls, loss_bbox, accuracy_cls


# ---------------------------------------------------------------------------- #
# Box heads
# ---------------------------------------------------------------------------- #

class roi_2mlp_head(nn.Module):
    """Add a ReLU MLP with two hidden layers."""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale
        self.dim_out = hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM

        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc1 = nn.Linear(dim_in * roi_size**2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        mynn.init.XavierFill(self.fc1.weight)
        init.constant_(self.fc1.bias, 0)
        mynn.init.XavierFill(self.fc2.weight)
        init.constant_(self.fc2.bias, 0)

    def detectron_weight_mapping(self):
        detectron_weight_mapping = {
            'fc1.weight': 'fc6_w',
            'fc1.bias': 'fc6_b',
            'fc2.weight': 'fc7_w',
            'fc2.bias': 'fc7_b'
        }
        return detectron_weight_mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = F.relu(self.fc1(x.view(batch_size, -1)), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        return x


class roi_Xconv1fc_head(nn.Module):
    """Add a X conv + 1fc head, as a reference if not using GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*2): 'head_conv%d_w' % (i+1),
                'convs.%d.bias' % (i*2): 'head_conv%d_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x


class roi_Xconv1fc_gn_head(nn.Module):
    """Add a X conv + 1fc head, with GroupNorm"""
    def __init__(self, dim_in, roi_xform_func, spatial_scale):
        super().__init__()
        self.dim_in = dim_in
        self.roi_xform = roi_xform_func
        self.spatial_scale = spatial_scale

        hidden_dim = cfg.FAST_RCNN.CONV_HEAD_DIM
        module_list = []
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            module_list.extend([
                nn.Conv2d(dim_in, hidden_dim, 3, 1, 1, bias=False),
                nn.GroupNorm(net_utils.get_group_gn(hidden_dim), hidden_dim,
                             eps=cfg.GROUP_NORM.EPSILON),
                nn.ReLU(inplace=True)
            ])
            dim_in = hidden_dim
        self.convs = nn.Sequential(*module_list)

        self.dim_out = fc_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
        roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
        self.fc = nn.Linear(dim_in * roi_size * roi_size, fc_dim)

        self._init_weights()

    def _init_weights(self):
        def _init(m):
            if isinstance(m, nn.Conv2d):
                mynn.init.MSRAFill(m.weight)
            elif isinstance(m, nn.Linear):
                mynn.init.XavierFill(m.weight)
                init.constant_(m.bias, 0)
        self.apply(_init)

    def detectron_weight_mapping(self):
        mapping = {}
        for i in range(cfg.FAST_RCNN.NUM_STACKED_CONVS):
            mapping.update({
                'convs.%d.weight' % (i*3): 'head_conv%d_w' % (i+1),
                'convs.%d.weight' % (i*3+1): 'head_conv%d_gn_s' % (i+1),
                'convs.%d.bias' % (i*3+1): 'head_conv%d_gn_b' % (i+1)
            })
        mapping.update({
            'fc.weight': 'fc6_w',
            'fc.bias': 'fc6_b'
        })
        return mapping, []

    def forward(self, x, rpn_ret):
        x = self.roi_xform(
            x, rpn_ret,
            blob_rois='rois',
            method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
            resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
            spatial_scale=self.spatial_scale,
            sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO
        )
        batch_size = x.size(0)
        x = self.convs(x)
        x = F.relu(self.fc(x.view(batch_size, -1)), inplace=True)
        return x
