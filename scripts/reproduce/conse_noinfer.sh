#! /bin/sh

CUDA_VISIBLE_DEVICES=0 python tools/test_net.py --dataset vgbansal --cfg configs/mine/rel_ft.yaml --set MODEL.NUM_CLASSES 609 MODEL.NUM_RELATIONS 21  REL_INFER.TRAIN True  TEST.TAGGING True TEST.USE_REL_INFER False TEST.SCORE_THRESH 0 MODEL.RELATION_NET_INPUT GEO REL_INFER.NONE_BGREL_WEIGHT 1 TEST.CONSE True --load_ckpt pretrained/relt_geo_sc_gcn.pth --output_dir_append conse_noinfer
python tools/show_result.py test/conse_noinfer/tagging_eval.pkl
