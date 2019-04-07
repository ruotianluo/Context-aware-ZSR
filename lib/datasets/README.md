Prepare the images and annotations as [link](https://github.com/ronghanghu/seg_every_thing/blob/master/lib/datasets/data/README.md)

then run

```
python lib/datasets/vg/step_1_build_vg_raw_json.py --output_dir ./data/vg150/ --num_objects 150 --output_json instances_vg150_raw.json
python lib/datasets/vg/step_2_build_vg_cocoaligned_json.py --vg_raw_json_file ./data/vg150/instances_vg150_raw.json --output_dir ./data/vg150/ --output_json instances_vg150_cocoaligned --onlycoco_nococo
```