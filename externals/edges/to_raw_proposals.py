import scipy.io as sio

img_list = open('vg_test_list.txt').readlines()

img_list = [_.split('.')[0] for _ in img_list]

proposals = sio.loadmat('bbs.mat')['bbs']
import torch
torch.save({k:v[0] for k,v in zip(img_list, proposals)}, '../../data/vg/bansal/raw_proposals.pth')
