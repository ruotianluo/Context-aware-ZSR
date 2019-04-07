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

"""Test a Detectron network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import cv2
import datetime
import logging
import numpy as np
import os
import yaml

import torch

from core.config import cfg
from core.rpn_generator import generate_rpn_on_dataset
from core.rpn_generator import generate_rpn_on_range
from core.test import im_detect_all
from datasets import task_evaluation
from datasets.json_dataset import JsonDataset
from modeling import model_builder
import nn as mynn
from utils.detectron_weight_helper import load_detectron_weight
import utils.env as envu
import utils.net as net_utils
import utils.subprocess as subprocess_utils
import utils.vis as vis_utils
from utils.io import save_object
from utils.timer import Timer

logger = logging.getLogger(__name__)


def get_eval_functions():
    # Determine which parent or child function should handle inference
    if False: #cfg.MODEL.RPN_ONLY or (not cfg.RPN.RPN_ON and cfg.TEST.PRECOMPUTED_PROPOSALS):
        child_func = generate_rpn_on_range
        parent_func = generate_rpn_on_dataset
    else:
        # Generic case that handles all network types other than RPN-only nets
        # and RetinaNet
        child_func = test_net
        parent_func = test_net_on_dataset

    return parent_func, child_func


def get_inference_dataset(index, is_parent=True):
    assert is_parent or len(cfg.TEST.DATASETS) == 1, \
        'The child inference process can only work on a single dataset'

    dataset_name = cfg.TEST.DATASETS[index]

    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert is_parent or len(cfg.TEST.PROPOSAL_FILES) == 1, \
            'The child inference process can only work on a single proposal file'
        assert len(cfg.TEST.PROPOSAL_FILES) == len(cfg.TEST.DATASETS), \
            'If proposals are used, one proposal file must be specified for ' \
            'each dataset'
        proposal_file = cfg.TEST.PROPOSAL_FILES[index]
    else:
        proposal_file = None

    return dataset_name, proposal_file


def run_inference(
        args, ind_range=None,
        multi_gpu_testing=False, gpu_id=0,
        check_expected_results=False):
    parent_func, child_func = get_eval_functions()
    is_parent = ind_range is None

    def result_getter():
        if True: #is_parent:
            # Parent case:
            # In this case we're either running inference on the entire dataset in a
            # single process or (if multi_gpu_testing is True) using this process to
            # launch subprocesses that each run inference on a range of the dataset
            all_results = {}
            for i in range(len(cfg.TEST.DATASETS)):
                dataset_name, proposal_file = get_inference_dataset(i)
                output_dir = args.output_dir
                results = parent_func(
                    args,
                    dataset_name,
                    proposal_file,
                    output_dir,
                    ind_range=ind_range,
                    multi_gpu=multi_gpu_testing
                )
                all_results.update(results)

            return all_results
        else:
            # Subprocess child case:
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset
            dataset_name, proposal_file = get_inference_dataset(0, is_parent=False)
            output_dir = args.output_dir
            return child_func(
                args,
                dataset_name,
                proposal_file,
                output_dir,
                ind_range=ind_range,
                gpu_id=gpu_id
            )

    all_results = result_getter()
    if check_expected_results and is_parent:
        task_evaluation.check_expected_results(
            all_results,
            atol=cfg.EXPECTED_RESULTS_ATOL,
            rtol=cfg.EXPECTED_RESULTS_RTOL
        )
        task_evaluation.log_copy_paste_friendly_results(all_results)

    return all_results


def test_net_on_dataset(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        multi_gpu=False,
        gpu_id=0):
    """Run inference on a dataset."""
    dataset = JsonDataset(dataset_name)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            args, dataset_name, proposal_file, num_images, output_dir
        )
    else:
        all_boxes, all_segms, all_keyps = test_net(
            args, dataset_name, proposal_file, output_dir, ind_range=ind_range, gpu_id=gpu_id
        )
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(test_timer.average_time))

    dataset.test_img_ids = sorted(dataset.COCO.getImgIds())
    if ind_range is not None:
        dataset.test_img_ids = dataset.test_img_ids[ind_range[0]:ind_range[1]]

    results = task_evaluation.evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir
    )
    return results


def multi_gpu_test_net_on_dataset(
        args, dataset_name, proposal_file, num_images, output_dir):
    """Multi-gpu inference on a dataset."""
    binary_dir = envu.get_runtime_dir()
    binary_ext = envu.get_py_bin_ext()
    binary = os.path.join(binary_dir, args.test_net_file + binary_ext)
    assert os.path.exists(binary), 'Binary \'{}\' not found'.format(binary)

    # Pass the target dataset and proposal file (if any) via the command line
    opts = ['TEST.DATASETS', '("{}",)'.format(dataset_name)]
    if proposal_file:
        opts += ['TEST.PROPOSAL_FILES', '("{}",)'.format(proposal_file)]

    # Run inference in parallel in subprocesses
    # Outputs will be a list of outputs from each subprocess, where the output
    # of each subprocess is the dictionary saved by test_net().
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir,
        args.load_ckpt, args.load_detectron, opts
    )

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for cls_idx in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[cls_idx] += all_boxes_batch[cls_idx]
            all_segms[cls_idx] += all_segms_batch[cls_idx]
            all_keyps[cls_idx] += all_keyps_batch[cls_idx]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net(
        args,
        dataset_name,
        proposal_file,
        output_dir,
        ind_range=None,
        gpu_id=0):
    """Run inference on all images in a dataset or over an index range of images
    in a dataset using a single GPU.
    """
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'

    roidb, dataset, start_ind, end_ind, total_num_images = get_roidb_and_dataset(
        dataset_name, proposal_file, ind_range
    )
    model = initialize_model_from_cfg(args, roidb = roidb, gpu_id=gpu_id)
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)

    if cfg.TEST.TAGGING:
        all_scores = []

    for i, entry in enumerate(roidb):
        if cfg.TEST.PRECOMPUTED_PROPOSALS:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue
        elif cfg.TEST.USE_GT_PROPOSALS:
            box_proposals = entry['boxes'][entry['gt_classes'] > 0]
        else:
            # Faster R-CNN type models generate proposals on-the-fly with an
            # in-network RPN; 1-stage models don't require proposals.
            box_proposals = None

        im = cv2.imread(entry['image'])
        if cfg.TEST.TAGGING:
            cls_boxes_i, cls_segms_i, cls_keyps_i, scores = im_detect_all(model, im, entry, box_proposals, timers)
            all_scores.append(scores)
        else:
            cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(model, im, entry, box_proposals, timers)

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (
                timers['im_detect_bbox'].average_time +
                timers['im_detect_mask'].average_time +
                timers['im_detect_keypoints'].average_time
            )
            misc_time = (
                timers['misc_bbox'].average_time +
                timers['misc_mask'].average_time +
                timers['misc_keypoints'].average_time
            )
            logger.info(
                (
                    'im_detect: range [{:d}, {:d}] of {:d}: '
                    '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})'
                ).format(
                    start_ind + 1, end_ind, total_num_images, start_ind + i + 1,
                    start_ind + num_images, det_time, misc_time, eta
                )
            )

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.vis_one_image(
                im[:, :, ::-1],
                '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'),
                cls_boxes_i,
                segms=cls_segms_i,
                keypoints=cls_keyps_i,
                thresh=cfg.VIS_TH,
                box_alpha=0.8,
                dataset=dataset,
                show_class=True
            )

    # Evaluate relations
    if cfg.TEST.EVALUATE_REL_ACC:
        from sklearn.metrics import precision_recall_curve
        scores_collect = np.vstack(model.module.Rel_Inf.rel_scores_collect)
        gt_collect = np.hstack(model.module.Rel_Inf.rel_gt_collect)
        recalls = []
        for i in range(scores_collect.shape[-1]):
            filt = gt_collect==i
            print(i, accuracy(torch.from_numpy(scores_collect[filt]), torch.from_numpy(gt_collect[filt]).long(), (1,2)))
            recalls.append(precision_recall_curve(gt_collect==i, scores_collect[:,i])[1].mean())
            print(i, recalls[-1])
            # how to get recall!
        import pdb;pdb.set_trace()
        print(np.array(recalls).mean())

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    save_object(
        dict(
            all_boxes=all_boxes,
            all_segms=all_segms,
            all_keyps=all_keyps,
            cfg=cfg_yaml,
            im_filenames=[entry['image'] for entry in roidb],
            classes=dataset.classes
        ), det_file
    )
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    if cfg.TEST.TAGGING:
        # Save results
        tagging_name = 'tagging.pkl' if ind_range is None else 'tagging_range_%s_%s.pkl' % tuple(ind_range)
        tagging_file = os.path.join(output_dir, tagging_name)
        
        save_object(
            dict(
                all_scores=all_scores,
                gt_classes=[r['gt_classes'] for r in roidb]
            ), tagging_file
        )
        logger.info('Wrote tagging results to: {}'.format(tagging_file))

        # Evaluate tagging
        all_scores = np.vstack(all_scores)
        gt_classes = np.hstack([r['gt_classes'] for r in roidb])
        img_id = \
            np.hstack([np.ones(len(roidb[i]['gt_classes']))*i for i in range(len(roidb))])

        tagging_eval = {}
        print('Compute AUSUC')
        tagging_eval['ausuc'] = Compute_AUSUC(dataset, all_scores, gt_classes, cfg.TEST.CLASS_SPLIT['source'], cfg.TEST.CLASS_SPLIT['target'])
        print('Generalized on all')
        tagging_eval['all'] = evaluate(dataset, all_scores, gt_classes)
        tagging_eval['all'].update(mean_img_eval(all_scores, gt_classes, img_id))
        # Generalized on source
        source_filter = np.isin(gt_classes, cfg.TEST.CLASS_SPLIT['source'])
        _all_scores, _gt_classes = all_scores[source_filter], gt_classes[source_filter]
        _img_id = img_id[source_filter]
        if source_filter.any(): # Only when there are source ground truth
            print('Generalized on source')
            tagging_eval['gen_source'] = evaluate(dataset, _all_scores, _gt_classes)
            tagging_eval['gen_source'].update(\
                mean_img_eval(_all_scores, _gt_classes, _img_id))
            # not generalized on source
            inf_scores = np.zeros(all_scores.shape[1])
            inf_scores[cfg.TEST.CLASS_SPLIT['target']] = float('-inf')
            _all_scores, _gt_classes = all_scores[source_filter]+inf_scores, gt_classes[source_filter]
            
            print('Ungeneralized on source')
            tagging_eval['ungen_source'] = evaluate(dataset, _all_scores, _gt_classes)
            tagging_eval['ungen_source'].update(\
                mean_img_eval(_all_scores, _gt_classes, _img_id))
        # The above is showing how target labels are confusing source boxes.
        # Generalized on target
        target_filter = np.isin(gt_classes, cfg.TEST.CLASS_SPLIT['target'])
        _all_scores, _gt_classes = all_scores[target_filter], gt_classes[target_filter]
        _img_id = img_id[target_filter]
        if target_filter.any(): # Only when there are target ground truth
            print('Generalized on target')
            tagging_eval['gen_target'] = evaluate(dataset, _all_scores, _gt_classes)
            tagging_eval['gen_target'].update(\
                mean_img_eval(_all_scores, _gt_classes, _img_id))
            # not geralized on target
            inf_scores = np.zeros(all_scores.shape[1])
            inf_scores[cfg.TEST.CLASS_SPLIT['source']] = float('-inf')
            _all_scores, _gt_classes = all_scores[target_filter]+inf_scores, gt_classes[target_filter]
            
            img_id = \
                np.hstack([\
                np.ones(np.isin(roidb[i]['gt_classes'], cfg.TEST.CLASS_SPLIT['source']).sum())\
                for i in range(len(roidb))])
                
            print('Ungeneralized on target')
            tagging_eval['ungen_target'] = evaluate(dataset, _all_scores, _gt_classes)
            tagging_eval['ungen_target'].update(\
                mean_img_eval(_all_scores, _gt_classes, _img_id))
        
        tagging_eval_name = 'tagging_eval.pkl' if ind_range is None else 'tagging_eval_range_%s_%s.pkl' % tuple(ind_range)
        tagging_eval_file = os.path.join(output_dir, tagging_eval_name)
        save_object(tagging_eval, tagging_eval_file)
        logger.info('Wrote tagging eval results to: {}'.format(tagging_eval_file))

    # # Pad None for standard coco evaluation
    # if ind_range is not None:
    #     for i in range(len(all_boxes)):
    #         all_boxes[i] = [None] * start_ind + all_boxes[i] + [None] * (total_num_images - end_id)

    return all_boxes, all_segms, all_keyps

def mean_img_eval(all_scores, gt_classes, img_id):
    num_images = int(img_id[-1]) + 1

    acs = [0.] * num_images
    acs_all = [0.] * num_images
    valid = [0] * num_images

    # Need to remove the background class
    max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1

    for i in range(num_images):
        ind_this = np.where(img_id == i)[0]
        acs_all[i] = np.sum(max_inds[ind_this] == gt_classes[ind_this])
        if ind_this.shape[0] > 0:
            valid[i] = ind_this.shape[0]
            acs[i] = acs_all[i] / ind_this.shape[0]

    mimg_ac = np.mean([s for s, v in zip(acs,valid) if v])
    # mimg_ac.per_img = [(s, v) for s, v in zip(acs,valid)]
    print(('mean-img: {:.3f}'.format(mimg_ac)))
    return {'img_ac':acs,  'mean_img': mimg_ac}


def Compute_AUSUC(dataset, all_scores, gt_classes, seen, unseen):
    cls_in_test = set(np.unique(gt_classes).tolist())
    seen = sorted(list(cls_in_test.intersection(set(seen))))
    unseen = sorted(list(cls_in_test.intersection(set(unseen))))
    score_S = all_scores[:, seen]
    score_U = all_scores[:, unseen]
    Y = gt_classes
    label_S = np.array(seen)
    label_U = np.array(unseen)

    AUC_val, AUC_record, acc_noBias, HM, fixed_bias = _Compute_AUSUC(
        torch.from_numpy(score_S),
        torch.from_numpy(score_U),
        torch.from_numpy(Y.astype(np.int64)),
        torch.from_numpy(label_S.astype(np.int64)),
        torch.from_numpy(label_U.astype(np.int64)))

    HM, fixed_bias = HM.item(), fixed_bias.item()
    print('AUC_val: {:.3f} HM: {:.3f} fixed_bias: {:.3f}'\
        .format(AUC_val, HM, fixed_bias))

    return {'AUC_val':AUC_val, 'AUC_record':AUC_record,\
        'acc_noBias': acc_noBias, 'HM': HM, 'fixed_bias': fixed_bias}

def _Compute_AUSUC(score_S, score_U, Y, label_S, label_U, fixed_bias = None):
    # % score_S: #samples-by-#seen classes (columns should correspond to seen-class labels in ascending order)
    # % score_U: #samples-by-#unseen classes (columns should correspond to unseen-class labels in ascending order)
    # % Y: #samples-by-1 vector
    # % label_S: #classes-by-1 vector (in ascending order)
    # % label_U: #classes-by-1 vector (in ascending order)
    # % fixed_bias: a scalar of bias to increase unseen classes' scores. If [],
    # % the fixed_bias that leads to the highest HM will be returned 

    unique = lambda x: torch.unique(x, True)
    torch.set_default_dtype(torch.double)
    # From harry chao's zero-shot-learning-journal repo
    def Compute_HM(acc):
        HM = 2 * acc[0] * acc[1] / (acc[0] + acc[1])
        return HM

    def AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Ytrue):

        L_S = len(label_S)
        L_U = len(label_U)
        class_count_S = torch.zeros(L_S)
        class_count_U = torch.zeros(L_U)
        class_correct_S = torch.zeros(L_S)
        class_correct_U = torch.zeros(L_U)
        for i in range(L_S):
            class_count_S[i] = (Ytrue == label_S[i]).sum()
            class_correct_S[i] = ((Ytrue == label_S[i]) & (Ypred_S == label_S[i])).sum()

        for i in range(L_U):
            class_count_U[i] = (Ytrue == label_U[i]).sum()
            class_correct_U[i] = ((Ytrue == label_U[i]) & (Ypred_U == label_U[i])).sum()

        class_count_S[class_count_S == 0] = 10 ** 10
        class_count_U[class_count_U == 0] = 10 ** 10
        return class_correct_S, class_correct_U, class_count_S, class_count_U

    AUC_record = torch.zeros(len(Y) + 1, 2)
    label_S = unique(label_S)
    label_U = unique(label_U)

    L_S = len(label_S)
    L_U = len(label_U)
    L = len(unique(torch.cat([label_S, label_U], 0)))
    if L_S + L_U != L or len(unique(Y)) != L:
        print('Wrong seen-unseen separation')
        import pdb;pdb.set_trace()
    if L_S != score_S.shape[1] or L_U != score_U.shape[1]:
        print('Wrong class number')
        import pdb;pdb.set_trace()

    # %% effective bias searching
    max_S, loc_S = score_S.max(1)
    Ypred_S = label_S[loc_S]
    max_U, loc_U = score_U.max(1)
    Ypred_U = label_U[loc_U]
    class_correct_S, class_correct_U, class_count_S, class_count_U = AUC_eval_class_count(Ypred_S, Ypred_U, label_S, label_U, Y)
    Y_correct_S = (Ypred_S == Y).double()
    Y_correct_U = (Ypred_U == Y).double()
    bias = max_S - max_U
    bias, loc_B = torch.sort(bias)
    unique_bias_loc = torch.unique(bias, True, return_inverse=True)[1];
    unique_bias_loc = unique_bias_loc[unique_bias_loc != 0];unique_bias_loc = unique_bias_loc - 1
    unique_bias_loc = unique(torch.tensor(unique_bias_loc.tolist()+ [len(bias)-1]))
    bias = bias[unique_bias_loc]
    # %% efficient evaluation
    acc_change_S = (Y_correct_S[loc_B] / class_count_S[loc_S[loc_B]]) / L_S
    acc_change_U = (Y_correct_U[loc_B] / class_count_U[loc_U[loc_B]]) / L_U
    AUC_record[:, 0] = torch.tensor([0]+torch.cumsum(-acc_change_S, 0).tolist()) + (class_correct_S / class_count_S).mean()
    AUC_record[:, 1] = torch.tensor([0]+torch.cumsum(acc_change_U, 0).tolist())

    if (AUC_record[-1] - torch.tensor([0, (class_correct_U / class_count_U).mean()])).abs().sum() > (10 ** -12):
        print('AUC wrong');
        import pdb;pdb.set_trace()
    AUC_record = AUC_record[[0] + unique_bias_loc.view(-1).tolist()]
    # %% Compute AUC
    acc_noBias = AUC_record[(bias <= 0).sum()]
    AUC_val = -np.trapz(AUC_record[:, 1].numpy(), AUC_record[:, 0].numpy())
    # %% Compute Harmonic mean
    if fixed_bias is None:
        HM = 2 * (AUC_record[:, 1] * AUC_record[:, 0]) / (AUC_record[:, 1] + AUC_record[:, 0])
        HM, fixed_bias_loc = torch.max(HM, 0)
        fixed_bias_loc = max(0, fixed_bias_loc - 1); # TODO ??
        fixed_bias = bias[fixed_bias_loc]
        print('Fix fixed bias:', fixed_bias)
    acc = AUC_record[torch.sum(bias <= fixed_bias)]
    HM = Compute_HM(acc)
    HM_nobias = Compute_HM(acc_noBias)
    print('without bias: acc_S: '+ str(acc_noBias[0])+'; acc_U: '+str(acc_noBias[1])+'; HM: '+str(HM_nobias))
    print('with bias '+str(fixed_bias)+': acc_S: '+str(acc[0])+'; acc_U: '+str(acc[1])+'; HM: '+str(HM))
    return AUC_val, AUC_record, acc_noBias, HM, fixed_bias


def evaluate(dataset, all_scores, gt_classes):
    # Precision
    precs = accuracy(torch.from_numpy(all_scores), torch.from_numpy(gt_classes).long(), (1,5,10,20))
    print('Prec@(1,5,10,20): %.3f\t%.3f\t%.3f\t%.3f' %(precs[0][0],precs[1][0],precs[2][0],precs[3][0]))

    # Copy from endernewton/iter-reason
    def voc_ap(rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def _score(dataset, all_scores, gt_classes):
        scs = [0.] * dataset.num_classes
        scs_all = [0.] * dataset.num_classes
        valid = [0] * dataset.num_classes
        for i in range(1, dataset.num_classes):
            ind_this = np.where(gt_classes == i)[0]  
            scs_all[i] = np.sum(all_scores[ind_this, i])
            if ind_this.shape[0] > 0:
                valid[i] = ind_this.shape[0]
                scs[i] = scs_all[i] / ind_this.shape[0]

        mcls_sc = np.mean([s for s, v in zip(scs,valid) if v])
        mins_sc = np.sum(scs_all) / gt_classes.shape[0]
        return scs[1:], mcls_sc, mins_sc, valid[1:]

    def _accuracy(dataset, all_scores, gt_classes):
        acs = [0.] * dataset.num_classes
        acs_all = [0.] * dataset.num_classes
        valid = [0] * dataset.num_classes

        # Need to remove the background class
        max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1
        max_scores = np.empty_like(all_scores)
        max_scores[:] = 0.
        max_scores[np.arange(gt_classes.shape[0]), max_inds] = 1.

        for i in range(1, dataset.num_classes):
            ind_this = np.where(gt_classes == i)[0]
            acs_all[i] = np.sum(max_scores[ind_this, i])
            if ind_this.shape[0] > 0:
                valid[i] = ind_this.shape[0]
                acs[i] = acs_all[i] / ind_this.shape[0]

        mcls_ac = np.mean([s for s, v in zip(acs,valid) if v])
        mins_ac = np.sum(acs_all) / gt_classes.shape[0]
        return acs[1:], mcls_ac, mins_ac

    def _average_precision(dataset, all_scores, gt_classes):
        aps = [0.] * dataset.num_classes
        valid = [0] * dataset.num_classes

        ind_all = np.arange(gt_classes.shape[0])
        num_cls = dataset.num_classes
        num_ins = ind_all.shape[0]

        for i, c in enumerate(dataset.classes):
            if i == 0:
                continue
            gt_this = (gt_classes == i).astype(np.float32)
            num_this = np.sum(gt_this)
            # if i % 10 == 0:
            #     print('AP for %s: %d/%d' % (c, i, num_cls))
            if num_this > 0:
                valid[i] = num_this
                sco_this = all_scores[ind_all, i]

                ind_sorted = np.argsort(-sco_this)

                tp = gt_this[ind_sorted]
                max_ind = num_ins - np.argmax(tp[::-1])
                tp = tp[:max_ind]
                fp = 1. - tp

                tp = np.cumsum(tp)
                fp = np.cumsum(fp)
                rec = tp / float(num_this)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

                aps[i] = voc_ap(rec, prec)

        mcls_ap = np.mean([s for s, v in zip(aps,valid) if v])

        # Compute the overall score
        max_inds = np.argmax(all_scores[:, 1:], axis=1) + 1
        max_scores = np.empty_like(all_scores)
        max_scores[:] = 0.
        max_scores[ind_all, max_inds] = 1.
        pred_all = max_scores[ind_all, gt_classes]
        sco_all = all_scores[ind_all, gt_classes]
        ind_sorted = np.argsort(-sco_all)

        tp = pred_all[ind_sorted]
        fp = 1. - tp

        tp = np.cumsum(tp)
        fp = np.cumsum(fp)
        rec = tp / float(num_ins)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        mins_ap = voc_ap(rec, prec)
        return aps[1:], mcls_ap, mins_ap

    scs, mcls_sc, mins_sc, valid = _score(dataset, all_scores, gt_classes)
    acs, mcls_ac, mins_ac = _accuracy(dataset, all_scores, gt_classes)
    aps, mcls_ap, mins_ap = _average_precision(dataset, all_scores, gt_classes)

    # for i, cls in enumerate(dataset.classes):
    #     if cls == '__background__' or not valid[i-1]:
    #         continue
    #     print(('{} {:d} {:.4f} {:.4f} {:.4f}'.format(cls, 
    #                                                 valid[i-1], 
    #                                                 scs[i-1], 
    #                                                 acs[i-1], 
    #                                                 aps[i-1])))

    # print('~~~~~~~~')
    # print('Scores | Accuracies | APs:')
    # for sc, ac, ap, vl in zip(scs, acs, aps, valid):
    #   if vl:
    #     print(('{:.3f} {:.3f} {:.3f}'.format(sc, ac, ap)))
    print('\t avg_score\tavg_accuracy\tavg_precision')
    print(('mean-cls: {:.3f} {:.3f} {:.3f}'.format(mcls_sc, mcls_ac, mcls_ap)))
    print(('mean-ins: {:.3f} {:.3f} {:.3f}'.format(mins_sc, mins_ac, mins_ap)))
    # print('~~~~~~~~')
    # print(('{:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}'.format(mcls_sc, 
    #                                                           mcls_ac, 
    #                                                           mcls_ap, 
    #                                                           mins_sc, 
    #                                                           mins_ac, 
    #                                                           mins_ap)))
    # print('~~~~~~~~')

    return {'precs':precs, 
            'mcls_ac':mcls_ac,
            'cls_ac':acs,
            'mcls_ap':mcls_ap,
            'mins_ac':mins_ac,
            'mins_ap':mins_ap}

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def initialize_model_from_cfg(args, roidb=None, gpu_id=0):
    """Initialize a model from the global cfg. Loads test-time weights and
    set to evaluation mode.
    """
    model = model_builder.Generalized_RCNN()
    model.eval()

    cfg.immutable(False)
    cfg.TEST.CLASS_SPLIT = {'source': roidb[0]['source'], 'target': roidb[0]['target']}
    cfg.immutable(True)

    if 'word_embeddings' in roidb[0]:
        model.Box_Outs.set_word_embedding(torch.tensor(roidb[0]['word_embeddings']))
    if cfg.MODEL.IGNORE_CLASSES:
        if cfg.MODEL.IGNORE_CLASSES == 'all':
            roidb[0]['all'] = roidb[0]['source'] + roidb[0]['target']
        model._ignore_classes = roidb[0][cfg.MODEL.IGNORE_CLASSES]
        model.Box_Outs._ignore_classes = roidb[0][cfg.MODEL.IGNORE_CLASSES]
    if True:
        tmp = {}
        for rel in roidb[0]['relationships']:
            tmp[(rel['subject_id'], rel['object_id'])] = \
                tmp.get((rel['subject_id'], rel['object_id']), []) + [rel['rel_id']]
        if cfg.MODEL.RELATION_COOCCUR:
            for k in tmp:
                tmp[k] = [1]
        if cfg.MODEL.NUM_RELATIONS > 0:
            model.Rel_Outs.relationship_dict = tmp

    if args.cuda:
        model.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        logger.info("loading checkpoint %s", load_name)
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(model, checkpoint['model'])

    if args.load_detectron:
        logger.info("loading detectron weights %s", args.load_detectron)
        load_detectron_weight(model, args.load_detectron)

    model = mynn.DataParallel(model, cpu_keywords=['im_info', 'roidb'], minibatch=True)

    return model


def get_roidb_and_dataset(dataset_name, proposal_file, ind_range):
    """Get the roidb for the dataset specified in the global cfg. Optionally
    restrict it to a range of indices if ind_range is a pair of integers.
    """
    dataset = JsonDataset(dataset_name)
    if cfg.TEST.PRECOMPUTED_PROPOSALS:
        assert proposal_file, 'No proposal file given'
        roidb = dataset.get_roidb(
            proposal_file=proposal_file,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT
        )
    else:
        roidb = dataset.get_roidb(gt=cfg.TEST.USE_GT_PROPOSALS)
    
    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end
    
    roidb[0].update({k:v for k,v in dataset.COCO.dataset.items() if k not in ['images', 'annotations', 'categories']})

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    """Return empty results lists for boxes, masks, and keypoints.
    Box detections are collected into:
      all_boxes[cls][image] = N x 5 array with columns (x1, y1, x2, y2, score)
    Instance mask predictions are collected into:
      all_segms[cls][image] = [...] list of COCO RLE encoded masks that are in
      1:1 correspondence with the boxes in all_boxes[cls][image]
    Keypoint predictions are collected into:
      all_keyps[cls][image] = [...] list of keypoints results, each encoded as
      a 3D array (#rois, 4, #keypoints) with the 4 rows corresponding to
      [x, y, logit, prob] (See: utils.keypoints.heatmaps_to_keypoints).
      Keypoints are recorded for person (cls = 1); they are in 1:1
      correspondence with the boxes in all_boxes[cls][image].
    """
    # Note: do not be tempted to use [[] * N], which gives N references to the
    # *same* empty list.
    all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)] for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    """Add results for an image to the set of all results at the specified
    index.
    """
    # Skip cls_idx 0 (__background__)
    for cls_idx in range(1, len(im_res)):
        all_res[cls_idx][index] = im_res[cls_idx]
