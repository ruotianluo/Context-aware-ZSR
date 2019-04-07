import scipy.io as sio

img_list = open('vg_test_list.txt').readlines()

img_list = [_.split('.')[0] for _ in img_list]

import json
test_images = json.load(open('/home-nfs/rluo/rluo/Detectron.pytorch/data/vgbansalnew/instances_vgbansalnew_test.json', 'r'))

proposals = []
bad_count = 0

img_list = [str(_['id']) for _ in test_images['images']]

for fn in img_list:
    try:
        proposals.append(sio.loadmat('edgebox2/'+fn+'.mat')['bbs'])
    except:
        proposals.append(0)
        bad_count += 1

print(bad_count)
print(len(proposals) - bad_count)

import torch
torch.save({k:v for k,v in zip(img_list, proposals)}, '/home-nfs/rluo/rluo/Detectron.pytorch/data/vg/bansal/raw_proposals.pth')


