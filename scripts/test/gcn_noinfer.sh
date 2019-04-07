#! /bin/sh

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset vgbansal --cfg configs/mine/rel_ft.yaml --set MODEL.NUM_CLASSES 609 MODEL.NUM_RELATIONS 21  REL_INFER.TRAIN True  TEST.TAGGING True TEST.USE_REL_INFER False TEST.SCORE_THRESH 0 MODEL.RELATION_NET_INPUT GEO --load_ckpt ./Outputs/rel_ft/ft_gt_relt_geo_sc_step/ckpt/model_final_gcn_wn.pth --output_dir_append gcn_noinfer
python tools/show_result.py test/gcn_noinfer/tagging_eval.pkl