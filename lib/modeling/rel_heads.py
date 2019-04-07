import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from core.config import cfg
import nn as mynn
import utils.net as net_utils
import numpy as np
import modeling.geo_feat as geo_feat

class rel_infer(nn.Module):
    def __init__(self, rel_out):
        super().__init__()
        self.rel_out = rel_out
        self.relationship_mat = None


        self.num_batches_tracked = 0
        self.avg_rel_tp = 0
        self.avg_rel_tn = 0
    
    def detectron_weight_mapping(self):
        return {}, []

    def init_rel_mat(self):
        self.relationship_dict = self.rel_out.relationship_dict
        coo = []
        for k,v in self.relationship_dict.items():
            for _ in v:
                coo.append([k[0], k[1], _])
        coo = torch.tensor(coo)
        self.relationship_mat = torch.sparse.FloatTensor(coo.t(), torch.ones(coo.shape[0]),
            torch.Size([cfg.MODEL.NUM_CLASSES,cfg.MODEL.NUM_CLASSES,cfg.MODEL.NUM_RELATIONS])).to_dense()
        if cfg.REL_INFER.CONSIDER_REVERSE_REL:
            self.relationship_mat = torch.cat([self.relationship_mat, self.relationship_mat.transpose(1,0)[...,1:]], 2)

    def forward(self, roi_scores, rel_scores, roidb):
        if roi_scores.shape[0] == 1 and roidb is not None and len(roidb[0]['boxes']) == 0:
            # The placeholder all zero rois, just return as it is.
            return roi_scores
        
        if roi_scores.shape[0] == 1:
            # Only one image, no need for propogation
            return roi_scores

        # Turn logprob to prob
        if not cfg.REL_INFER.TRAIN or cfg.REL_INFER.MODE == 2:
            rel_scores = rel_scores.exp()
    
        # Replace rel_scores with ground truth relationship if USE_GT_REL is True
        if cfg.TEST.USE_GT_REL or cfg.TEST.EVALUATE_REL_ACC:
            _rel_scores = torch.zeros_like(rel_scores).reshape(roi_scores.shape[0], roi_scores.shape[0], -1)
            for i in range(roi_scores.shape[0]):
                for j in range(roi_scores.shape[0]):
                    rel_ids = roidb[0]['gt_relationships'].get((i, j), [])
                    if cfg.MODEL.RELATION_COOCCUR and len(rel_ids) > 0:
                        rel_ids = [1]
                    if len(rel_ids) > 0:
                        _rel_scores[i][j][rel_ids] = 1 / len(rel_ids)
                    else:
                        _rel_scores[i][j][0] = 1
            _rel_scores = _rel_scores.view_as(rel_scores) 
        
        if cfg.TEST.EVALUATE_REL_ACC:
            filt = (_rel_scores > 0).float().sum(1) <= 1
            filt = (torch.eye(roi_scores.shape[0]).type_as(filt) != 1).view(-1) * filt
            if not hasattr(self, 'rel_scores_collect'):
                self.rel_scores_collect = []
                self.rel_gt_collect = []
            self.rel_scores_collect.append(rel_scores[filt].cpu().data.numpy())
            self.rel_gt_collect.append(_rel_scores[filt].max(1)[1].cpu().data.numpy())

            tmp_tp = (((rel_scores.max(1)[1] == _rel_scores.max(1)[1]) * (_rel_scores[:,1:] == 1).any(1)).float().sum() / ((_rel_scores[:, 1:] == 1).any(1).float().sum()+1e-7)).item()
            tmp_tn = (rel_scores.max(1)[1] == 0)[(_rel_scores[:,1:] == 0).all(1)].float().mean().item()
            print('Relation true positive acc', tmp_tp)
            print('Relation true negative acc', tmp_tn)
            self.avg_rel_tp = (self.avg_rel_tp * self.num_batches_tracked + tmp_tp) / (self.num_batches_tracked + 1)
            self.avg_rel_tn = (self.avg_rel_tn * self.num_batches_tracked + tmp_tn) / (self.num_batches_tracked + 1)
            self.num_batches_tracked += 1
            print('Relation avg tp acc', self.avg_rel_tp)
            print('Relation avg tn acc', self.avg_rel_tn)
        
        if cfg.TEST.USE_GT_REL:
            rel_scores = _rel_scores

        if self.relationship_mat is None:
            self.init_rel_mat()
            self.relationship_mat = self.relationship_mat.type_as(roi_scores)
            self.relationship_mat[:,:,0].fill_(1)

        assert roi_scores.shape[0] ** 2 == rel_scores.shape[0]
        rel_scores = rel_scores.reshape(roi_scores.shape[0], roi_scores.shape[0], -1)

        # Take topk to reduce search space. k = 5, so we can perverse recall@5
        topk_scores, topk_labels = roi_scores.topk(cfg.TEST.REL_INFER_PROPOSAL, 1)
        
        # Approximate Q
        Q = torch.ones_like(topk_scores)
        Q = F.softmax(Q) # normalize

        # Get binary terms, log energy.
        tmp_index = topk_labels.view(-1)
        tmp_index = torch.cat([tmp_index.view(-1, 1).repeat(1, tmp_index.shape[0]).view(1, -1),
                                tmp_index.view(1, -1).repeat(tmp_index.shape[0], 1).view(1, -1)], 0)
        binary_theta = self.relationship_mat[tmp_index[0], tmp_index[1]].view(*(topk_scores.shape[:2]+topk_scores.shape[:2]+(-1,)))
        if not cfg.REL_INFER.NO_REL_SCORE:
            binary_theta *= rel_scores.unsqueeze(1).unsqueeze(3)
        if cfg.REL_INFER.MODE == 3:
            binary_theta = binary_theta[..., 0] + cfg.REL_INFER.NONE_BGREL_WEIGHT * binary_theta[..., 1:].max(-1)[0] # KxNxN
        else:
            binary_theta = binary_theta[..., 0] + cfg.REL_INFER.NONE_BGREL_WEIGHT * binary_theta[..., 1:].sum(-1) # KxNxN
        # make binary_theta symmetric
        binary_theta = (binary_theta + binary_theta.permute(2,3,0,1))/2
        # Take log
        if not cfg.REL_INFER.TRAIN or cfg.REL_INFER.MODE == 2:
            binary_theta = binary_theta.log() 
        # Make diagonal zero, cause the message pass only from neighbours
        torch.diagonal(binary_theta, dim1=0, dim2=2).fill_(0)

        if cfg.REL_INFER.BINARY_NORMALIZE:
            weight = rel_scores[:,:,1:].sum(-1)
            weight = (weight + weight.t()) / 2
            torch.diagonal(weight).fill_(0)
            weight = weight / weight.sum(-1, keepdim=True)
            binary_theta = binary_theta * weight.unsqueeze(-1).unsqueeze(1)

        # Get unary_theta
        unary_theta = topk_scores.log()

        for r in range(cfg.TEST.REL_INFER_ROUND):
            new_Q = cfg.TEST.REL_INFER_BINARY_WEIGHT * binary_theta.view(Q.numel(), -1).matmul(Q.view(-1, 1)).view_as(Q)
            new_Q = new_Q + unary_theta
            new_Q = F.softmax(new_Q)
            Q = new_Q

        if torch.isnan(Q).any():
            import pdb;pdb.set_trace()

        new_roi_scores = torch.zeros_like(roi_scores)
        new_roi_scores.scatter_(1, topk_labels, Q)
        roi_scores = (roi_scores + 10000 * new_roi_scores) / (1+10000)

        return roi_scores

class rel_infer_train(nn.Module):
    def __init__(self, rel_out):
        super().__init__()
        self.rel_out = rel_out
        self.relationship_mat = None


        self.num_batches_tracked = 0
        self.avg_rel_tp = 0
        self.avg_rel_tn = 0
    
    def detectron_weight_mapping(self):
        return {}, []

    def init_rel_mat(self):
        self.relationship_dict = self.rel_out.relationship_dict
        coo = []
        for k,v in self.relationship_dict.items():
            for _ in v:
                coo.append([k[0], k[1], _])
        coo = torch.tensor(coo)
        self.relationship_mat = torch.sparse.FloatTensor(coo.t(), torch.ones(coo.shape[0]),
            torch.Size([cfg.MODEL.NUM_CLASSES,cfg.MODEL.NUM_CLASSES,cfg.MODEL.NUM_RELATIONS])).to_dense()

        if cfg.REL_INFER.CONSIDER_REVERSE_REL:
            self.relationship_mat = torch.cat([self.relationship_mat, self.relationship_mat.transpose(1,0)[...,1:]], 2)

    def forward(self, rois, roi_labels, roi_scores, rel_scores, roidb):

        # Remove roi_labels being zero
        if roi_labels is not None:
            roi_scores = roi_scores[torch.from_numpy(roi_labels) > 0]
            rois = rois[roi_labels > 0]
            roi_labels = roi_labels[roi_labels > 0]
        proposal_num = get_proposal_num(rois) # Get the numbers of each image

        roi_labels = torch.from_numpy(roi_labels.astype('int64')).to(roi_scores.device)

        if self.relationship_mat is None:
            self.init_rel_mat()
            self.relationship_mat = self.relationship_mat.type_as(roi_scores)
            self.relationship_mat[:,:,0].fill_(1)

        roi_scores = F.log_softmax(roi_scores, dim=1)

        # Turn logprob to prob
        if not cfg.REL_INFER.TRAIN or cfg.REL_INFER.MODE == 2:
            rel_scores = rel_scores.exp()

        assert(sum([_**2 for _ in proposal_num]) == rel_scores.shape[0])

        head_rois = 0
        head_out = 0

        loss = []
        for i in range(len(proposal_num)):
            # Generate sub_obj features
            if proposal_num[i] == 0:
                continue

            lc_roi_labels = roi_labels[head_rois:head_rois+proposal_num[i]] # N
            lc_roi_scores = roi_scores[head_rois:head_rois+proposal_num[i]] # NxC
            lc_rel_scores = rel_scores[head_out:head_out+proposal_num[i]**2].reshape(proposal_num[i], proposal_num[i], -1)

            # import pdb;pdb.set_trace()
            binary_theta = self.relationship_mat[lc_roi_labels].unsqueeze(1).repeat(1, proposal_num[i],1,1)
            binary_theta = binary_theta * lc_rel_scores.unsqueeze(-2) # NxNx600xR
            
            if cfg.REL_INFER.MODE == 3:
                binary_theta = binary_theta[..., 0] + cfg.REL_INFER.NONE_BGREL_WEIGHT * binary_theta[..., 1:].max(-1)[0] # KxNxN
            else:
                binary_theta = binary_theta[..., 0] + cfg.REL_INFER.NONE_BGREL_WEIGHT * binary_theta[..., 1:].sum(-1) # KxNxN
            
            # t_binary the same as above
            t_binary_theta = self.relationship_mat[:, lc_roi_labels] # 600xNxR
            t_binary_theta = self.relationship_mat[:, lc_roi_labels].transpose(1,0).unsqueeze(1).repeat(1, proposal_num[i],1,1)

            t_binary_theta = t_binary_theta * lc_rel_scores.transpose(0,1).unsqueeze(-2) # NxNxKxR

            t_binary_theta = t_binary_theta[..., 0] + cfg.REL_INFER.NONE_BGREL_WEIGHT * t_binary_theta[..., 1:].sum(-1) # KxNxN

            # make binary_theta symmetric
            binary_theta = (binary_theta + t_binary_theta)/2 # NxNxK

            # remove diagonal
            tmp_diag = 1-torch.eye(proposal_num[i]).type_as(binary_theta).unsqueeze(-1)
            binary_theta = binary_theta * tmp_diag

            binary_theta = binary_theta.sum(0) # Nxk
            # Turn logprob to prob
            if not cfg.REL_INFER.TRAIN or cfg.REL_INFER.MODE == 2:
                binary_theta = binary_theta.log()
            
            unary_theta = lc_roi_scores

            # _loss = -F.log_softmax(unary_theta + binary_theta).gather(1, lc_roi_labels.unsqueeze(1)).squeeze(1)
            _loss = -F.log_softmax(binary_theta).gather(1, lc_roi_labels.unsqueeze(1)).squeeze(1)
            loss.append(_loss)

            head_out += proposal_num[i]**2
            head_rois += proposal_num[i]

            # if torch.isnan(loss[-1]):
            #     import pdb;pdb.set_trace()
        if len(loss) > 0:
            return torch.cat(loss, 0).mean()
        else:
            return roi_scores.new_tensor(0)

class rel_outputs(nn.Module):
    def __init__(self, dim_in):
        super().__init__()
        if cfg.REL_INFER.PRETRAINED_WE:
            out_rel_dim = 300
        elif not cfg.REL_INFER.CONSIDER_REVERSE_REL:
            out_rel_dim = cfg.MODEL.NUM_RELATIONS
        else:
            out_rel_dim = cfg.MODEL.NUM_RELATIONS * 2 -1

        if not cfg.MODEL.RELATION_NET_INPUT.startswith('GEO'):
            self.bilinear_sub = nn.Sequential(
                nn.Linear(dim_in, 32),
                nn.ReLU(inplace=True)
            )
            self.bilinear_obj = nn.Sequential(
                nn.Linear(dim_in, 32),
                nn.ReLU(inplace=True)
            )
        
        if cfg.MODEL.RELATION_NET_INPUT.startswith('NEW+GEO'):
            self.embed_geo = nn.Sequential(
                nn.Linear(18 if 'GEO2' in cfg.MODEL.RELATION_NET_INPUT else 64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, out_rel_dim)
            )
            self.embed_appr = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, out_rel_dim)
            )
            return

        if 'GEO2' in cfg.MODEL.RELATION_NET_INPUT:
            self.bilinear_geo = nn.Sequential(
                nn.Linear(18, 32),
                nn.ReLU(inplace=True)
            )
        elif 'GEO' in cfg.MODEL.RELATION_NET_INPUT:
            self.bilinear_geo = nn.Sequential(
                nn.Linear(64, 32),
                nn.ReLU(inplace=True)
            )
        
        if cfg.MODEL.RELATION_NET_INPUT.startswith('GEO'):
            self.rel_dim_in = 32
        elif cfg.MODEL.RELATION_NET_INPUT.startswith('+GEO'):
            self.rel_dim_in = 64 + 32
        else:
            self.rel_dim_in = 64
        self.relation_net = nn.Sequential(
            nn.Linear(self.rel_dim_in, 64),
            nn.ReLU(),
            nn.Linear(64, out_rel_dim)
        )


    def set_word_embedding(self, embedding):
        if embedding.shape[0] != cfg.MODEL.NUM_RELATIONS - 1:
            print('WARNING: Word embedding size not matched')
            return
        self.word_embedding[0].copy_(embedding.mean(0))
        self.word_embedding[1:].copy_(embedding)
        self.word_embedding.copy_(F.normalize(self.cls_score.word_embedding, dim=1))

    def detectron_weight_mapping(self):
        return {}, []

    def forward(self, *args, **kwargs):
        if cfg.MODEL.RELATION_NET_INPUT.startswith('NEW+GEO'):
            return self._forward_new(*args, **kwargs)
        else:
            return self._forward(*args, **kwargs)

    def get_geo_feat(self, rois, roi_feats):
        if 'GEO2' in cfg.MODEL.RELATION_NET_INPUT:
            tmp_geo_feat = geo_feat.get_proposal_feat(torch.from_numpy(rois).to(roi_feats))
        elif 'GEO' in cfg.MODEL.RELATION_NET_INPUT:
            tmp_geo_feat = geo_feat.extract_multi_position_matrix_nd(torch.from_numpy(rois).to(roi_feats))
            tmp_geo_feat = geo_feat.extract_pairwise_multi_position_embedding_nd(tmp_geo_feat)
        return tmp_geo_feat
                

    def _forward(self, rois, roi_feats, roi_labels = None, roidb=None):
        """
        input:
        rois, rois_feats
        """
        if roi_feats.dim() == 4:
            roi_feats = roi_feats.squeeze(3).squeeze(2)

        if cfg.MODEL.RELATION_NET_FEAT_STOP_GRAG:
            roi_feats = roi_feats.detach()

        # Remove roi_labels being zero
        if roi_labels is not None:
            old_rois = rois
            old_roi_labels = roi_labels

            roi_feats = roi_feats[torch.from_numpy(roi_labels) > 0]
            rois = rois[roi_labels > 0]
            roi_labels = roi_labels[roi_labels > 0]

        proposal_num = get_proposal_num(rois) # Get the numbers of each image

        if not cfg.MODEL.RELATION_NET_INPUT.startswith('GEO'):
            sub_feats = self.bilinear_sub(roi_feats)
            obj_feats = self.bilinear_obj(roi_feats)

        total_pairs = sum([_**2 for _ in proposal_num])
        pairs = np.zeros((total_pairs, 2))
        rel_feats = roi_feats.new_zeros((total_pairs, self.rel_dim_in))

        head_rois = 0
        head_out = 0
        for i in range(len(proposal_num)):

            # Generate sub_obj features
            if proposal_num[i] == 0:
                continue
            if not cfg.MODEL.RELATION_NET_INPUT.startswith('GEO'):
                tmp_sub = sub_feats[head_rois:head_rois+proposal_num[i]]
                tmp_obj = obj_feats[head_rois:head_rois+proposal_num[i]]
                tmp_sub = tmp_sub.unsqueeze(1).expand(proposal_num[i], proposal_num[i], -1)
                tmp_obj = tmp_obj.unsqueeze(0).expand(proposal_num[i], proposal_num[i], -1)
                # tmp appearance feature
                tmp_appr_feat = torch.cat([tmp_sub, tmp_obj], 2).reshape(proposal_num[i]**2, -1)
            if 'GEO' in cfg.MODEL.RELATION_NET_INPUT:
                tmp_geo_feat = self.get_geo_feat(rois[head_rois:head_rois+proposal_num[i], 1:], roi_feats)
                tmp_geo_feat = self.bilinear_geo(tmp_geo_feat)
                tmp_geo_feat = tmp_geo_feat.view(-1, tmp_geo_feat.shape[-1])

            if cfg.MODEL.RELATION_NET_INPUT.startswith('GEO'):
                rel_feats[head_out:head_out+proposal_num[i]**2] = tmp_geo_feat
            elif cfg.MODEL.RELATION_NET_INPUT.startswith('+GEO'):
                rel_feats[head_out:head_out+proposal_num[i]**2,:-32] = tmp_appr_feat
                rel_feats[head_out:head_out+proposal_num[i]**2,-32:] = tmp_geo_feat
            else:
                rel_feats[head_out:head_out+proposal_num[i]**2] = tmp_appr_feat

            if roi_labels is not None:
                # Generate sub obj pairs
                tmp_lbl = torch.from_numpy(roi_labels[head_rois:head_rois+proposal_num[i]]).unsqueeze(1)
                tmp_pairs = torch.cat([
                    tmp_lbl.unsqueeze(1).expand(proposal_num[i], proposal_num[i], -1),\
                    tmp_lbl.unsqueeze(0).expand(proposal_num[i], proposal_num[i], -1)\
                    ], 2).numpy()
                # Manually set diagonal to 0,0
                np.fill_diagonal(tmp_pairs[:,:,0], -1)
                np.fill_diagonal(tmp_pairs[:,:,1], -1)
                
                pairs[head_out:head_out+proposal_num[i]**2] = np.reshape(tmp_pairs, (proposal_num[i]**2, -1))
            
            head_out += proposal_num[i]**2
            head_rois += proposal_num[i]
        logit_rel = self.relation_net(rel_feats)

        def saveit(logit):
            import glob
            import os
            sfiles = glob.glob('rel/*.pth')
            sfiles.sort(key=os.path.getmtime)
            if len(sfiles) == 0:
                torch.save(logit, 'rel/0.pth')
            else:
                idx = int(sfiles[-1].split('.')[0].split('/')[1])
                torch.save(logit, 'rel/%d.pth' %(idx+1))
        # saveit(logit_rel)

        if cfg.REL_INFER.PRETRAINED_WE:
            logit_rel = logit_rel.matmul(self.word_embedding)
        if not cfg.REL_INFER.TRAIN or cfg.REL_INFER.MODE == 2:
            logit_rel = F.log_softmax(logit_rel, dim=1)

        if roi_labels is not None:
            if cfg.TRAIN.USE_GT_REL:
                rel_labels = generate_gt_rel_labels(old_rois, old_roi_labels, roi_feats, pairs, roidb)
            else:
                if len(pairs) == 0:
                    return logit_rel, None
                rel_labels = generate_labels(pairs, self.relationship_dict)
            return logit_rel, rel_labels
        else:
            return logit_rel


    def _forward_new(self, rois, roi_feats, roi_labels = None, roidb=None):
        """
        input:
        rois, rois_feats
        """
        if roi_feats.dim() == 4:
            roi_feats = roi_feats.squeeze(3).squeeze(2)

        if cfg.MODEL.RELATION_NET_FEAT_STOP_GRAG:
            roi_feats = roi_feats.detach()

        # Remove roi_labels being zero
        if roi_labels is not None:
            old_rois = rois
            old_roi_labels = roi_labels

            roi_feats = roi_feats[torch.from_numpy(roi_labels) > 0]
            rois = rois[roi_labels > 0]
            roi_labels = roi_labels[roi_labels > 0]

        proposal_num = get_proposal_num(rois) # Get the numbers of each image

        sub_feats = self.bilinear_sub(roi_feats)
        obj_feats = self.bilinear_obj(roi_feats)

        total_pairs = sum([_**2 for _ in proposal_num])
        pairs = np.zeros((total_pairs, 2))
        rel_feats = roi_feats.new_zeros((total_pairs, 64))
        geo_feats = roi_feats.new_zeros((total_pairs, 64))


        head_rois = 0
        head_out = 0
        for i in range(len(proposal_num)):

            # Generate sub_obj features
            if proposal_num[i] == 0:
                continue
            tmp_sub = sub_feats[head_rois:head_rois+proposal_num[i]]
            tmp_obj = obj_feats[head_rois:head_rois+proposal_num[i]]
            tmp_sub = tmp_sub.unsqueeze(1).expand(proposal_num[i], proposal_num[i], -1)
            tmp_obj = tmp_obj.unsqueeze(0).expand(proposal_num[i], proposal_num[i], -1)
            # tmp appearance feature
            tmp_appr_feat = torch.cat([tmp_sub, tmp_obj], 2).reshape(proposal_num[i]**2, -1)
            
            tmp_geo_feat = self.get_geo_feat(rois[head_rois:head_rois+proposal_num[i], 1:], roi_feats)
            tmp_geo_feat = tmp_geo_feat.view(-1, tmp_geo_feat.shape[-1])

            rel_feats[head_out:head_out+proposal_num[i]**2] = tmp_appr_feat
            geo_feats[head_out:head_out+proposal_num[i]**2] = tmp_geo_feat

            if roi_labels is not None:
                # Generate sub obj pairs
                tmp_lbl = torch.from_numpy(roi_labels[head_rois:head_rois+proposal_num[i]]).unsqueeze(1)
                tmp_pairs = torch.cat([
                    tmp_lbl.unsqueeze(1).expand(proposal_num[i], proposal_num[i], -1),\
                    tmp_lbl.unsqueeze(0).expand(proposal_num[i], proposal_num[i], -1)\
                    ], 2).numpy()
                # Manually set diagonal to 0,0
                np.fill_diagonal(tmp_pairs[:,:,0], -1)
                np.fill_diagonal(tmp_pairs[:,:,1], -1)
                
                pairs[head_out:head_out+proposal_num[i]**2] = np.reshape(tmp_pairs, (proposal_num[i]**2, -1))
            
            head_out += proposal_num[i]**2
            head_rois += proposal_num[i]

        logit_rel = self.embed_geo(geo_feats) + self.embed_appr(rel_feats)
        if cfg.REL_INFER.PRETRAINED_WE:
            logit_rel = logit_rel.matmul(self.word_embedding)
        if not cfg.REL_INFER.TRAIN or cfg.REL_INFER.MODE == 2:
            logit_rel = F.log_softmax(logit_rel, dim=1)
        
        if roi_labels is not None:
            if cfg.TRAIN.USE_GT_REL:
                rel_labels = generate_gt_rel_labels(old_rois, old_roi_labels, roi_feats, pairs, roidb)
            else:
                rel_labels = generate_labels(pairs, self.relationship_dict)
            return logit_rel, rel_labels
        else:
            return logit_rel


from multiprocessing.dummy import Pool as ThreadPool 

def generate_labels(pairs, rel_dict):
    pool = ThreadPool(8)
    def func(pair):
        tmp = rel_dict.get((pair[0], pair[1]), [0])
        out = np.zeros(cfg.MODEL.NUM_RELATIONS, dtype=np.int32)
        if pair[0] != -1 and pair[1] != -1: # If pair = (-1,-1) then this pair is diagonal pair, we don't need the labels
            out[tmp] = 1
        return out
    results = pool.map(func, pairs)
    pool.close()
    pool.join()
    results = np.stack(results)
    return results

def generate_gt_rel_labels(rois, roi_labels, roi_feats, pairs, roidb):
    # rois and roi_labels has to be original
    proposal_num = get_proposal_num(rois)
    head_pointer = 0
    rel_labels = []
    for i in range(len(proposal_num)):
        if proposal_num[i] == 0:
            continue
        tmp_rel_labels = torch.zeros(proposal_num[i], proposal_num[i], cfg.MODEL.NUM_RELATIONS).long()
        tmp_rel_labels[:,:,0].fill_(1)
        for rel, rel_ids in roidb[i]['gt_relationships'].items():
            tmp_rel_labels[rel[0],rel[1],0] = 0
            tmp_rel_labels[rel[0],rel[1]][rel_ids] = 1
        # remove diagonals
        torch.diagonal(tmp_rel_labels, dim1=0, dim2=1).fill_(0)
        # Remove roi_label == 0 ones
        remaining_index = roi_labels[head_pointer: head_pointer + proposal_num[i]] > 0
        remaining_index = torch.from_numpy(remaining_index.astype('uint8'))
        if remaining_index.sum() >= 0:
            tmp_rel_labels = tmp_rel_labels[remaining_index][:, remaining_index]
            rel_labels.append(tmp_rel_labels.reshape(-1, cfg.MODEL.NUM_RELATIONS))
        head_pointer += proposal_num[i]
    
    out = torch.cat(rel_labels, 0).numpy()
    if out.shape[0] != pairs.shape[0]:
        import pdb;pdb.set_trace()
    return out


def get_proposal_num(rois):
    out = []
    for i in range(cfg.TRAIN.IMS_PER_BATCH):
        out.append(int((rois[:, 0] == i).sum()))
    return out