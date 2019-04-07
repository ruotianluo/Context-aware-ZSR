# Introduction
This repository includes the code for paper Context-aware zero-shot recognition.

The code is highly based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Thanks for the effort from original author.

# Requirement

- Pytorch 1.0
- Python 2 and python 3
- torchtext
- nltk
- pycocotools

# Preparation

## Download dataset

```
bash data/vg/download.sh
```

## Convert from bansal train test split.

```
(In python 2)
python lib/datasets/vg/convert_from_bansal.py
```

## Build necessary libs

```
cd lib
bash make.sh
cd ..
```

## Download pretrained imagenet-model

```
python tools/download_imagenet_weights.py
```

# To reproduce numbers in the paper

Download pretrained model from [link](https://drive.google.com/open?id=1KkXXgYU9nb9DBOYuDfPEDnJwi4IBEUcV).

run scripts in `scripts/reproduce`

# Training

## Train without unseen category annotations

```
bash scripts/train.sh
```

In this case, --cfg defines the general training test configuration, like roi align size, backbone bone etc.

--auto_resume enables it to resume from the latest snapshot, this has to be used with --id together.

--set overrides the items in the configuration files. Here we manually set NUM_CLASSES BASE_LR and MODEL.TAGGING.

--id allow us to identify where we save our model much easier.

iter_reason_new.yaml is borrowed from xinlei's iterative reasoning paper.

## Train with word embedding as last layer

```
bash scripts/train_we.sh
```

This append `MODEL.WORD_EMBEDDING_REGU True` in '--set'.

Normally, the last fc layer of fast rcnn head is 2048x601, where here this layer is replaced by two linear layer 2048x300 and 300x601. The weight of 300x601 layer is the word embedding of all the classes.

## Train with relation inference model.
```
bash train_rel.sh (bash train_we_rel.sh)
```

# Test in region classification setting

## Test with WE

```
bash scripts/test/we_infer.sh
bash scripts/test/we_noinfer.sh
```

## Test with conse

```
bash scripts/test/conse_infer.sh
bash scripts/test/conse_noinfer.sh
```

## Test with gcn
Download [zsl-pth-gcn](https://github.com/ruotianluo/zsl-gcn-pth) to `./externals`.

```
(In python 2)
python lib/zsl-gcn/convert_gcn.py --weight_folder ./Outputs/rel_ft/ft_gt_relt_geo_sc
```

```
bash scripts/test/gcn_infer.sh
bash scripts/test/gcn_noinfer.sh
```


## Test with sync

```
(In python 2)
python lib/zsl-gcn/convert_sync.py --weight_folder ./Outputs/iter_reason_new/irn_vg_gt_sc_step --dataset vgbansal --to_weight Outputs/rel_ft/ft_gt_relt_geo_sc_step/ckpt/model_final.pth
```



# Test in detection setting:

## Get edgebox proposals

```
cd ./external/edges
get_proposals.m
python to_raw_proposals.py
cd ../../data/vg/bansal/
python generate_proposals.py
```

or download from [link](https://drive.google.com/open?id=1CBu0AfXDUgDKW1Jj424ndKugQL_TAo5t), and put it under `./data/vg/bansal`

## Test

```
bash scripts/test/detection_gcn.sh
bash scripts/test/detection_sync.sh
```