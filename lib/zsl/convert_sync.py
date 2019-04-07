# problem, during test, the corresp should also be changed..
import argparse

from nltk.corpus import wordnet as wn
import re

import torch
import pickle
import json
import numpy as np
import torch.nn.functional as F

import os


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vgbansal', type=str, help='vgzs')
parser.add_argument('--weight_fn', default='ckpt/model_final.pth', type=str)
parser.add_argument('--weight_folder', default='./Outputs/iter_reason_new/rluo_irnb_vg_gt_cooc_pgeo_sc_step', type=str)
parser.add_argument('--to_weight', default=None, type=str)
parser.add_argument('--hiddens', default='--hiddens 2048d,d', type=str)

args = parser.parse_args()

# vg_dataset = 'vgzs'
vg_dataset = args.dataset

json_data = json.load(open('./data/%s/instances_%s_raw.json' %(vg_dataset, vg_dataset)))
weight_fn = os.path.join(args.weight_folder, args.weight_fn)

weight = torch.load(weight_fn)

#---------------------


from torchtext.vocab import GloVe

word_vectors = GloVe('6B')
feat_len = 300
def get_synset_embedding(synset, word_vectors, get_vector):
    class_name = wn.synset(synset).lemma_names()
    class_name = ', '.join([_.replace('_', ' ') for _ in class_name])
    class_name = class_name.lower()

    feat = np.zeros(feat_len)

    options = class_name.split(',')
    cnt_word = 0
    for j in range(len(options)):
        now_feat = get_embedding(options[j].strip(), word_vectors, get_vector)
        if np.abs(now_feat.sum()) > 0:
            cnt_word += 1
            feat += now_feat
    if cnt_word > 0:
        feat = feat / cnt_word

    if np.abs(feat.sum()) == 0:
        return feat
    else:
        # feat = feat / (np.linalg.norm(feat) + 1e-6)
        return feat

def get_embedding(entity_str, word_vectors, get_vector):
    try:
        feat = get_vector(word_vectors, entity_str)
        return feat
    except:
        feat = np.zeros(feat_len)

    str_set = list(filter(None, re.split("[ \-_]+", entity_str)))

    cnt_word = 0
    for i in range(len(str_set)):
        temp_str = str_set[i]
        try:
            now_feat = get_vector(word_vectors, temp_str)
            feat = feat + now_feat
            cnt_word = cnt_word + 1
        except:
            continue

    if cnt_word > 0:
        feat = feat / cnt_word
    return feat

def get_vector(word_vectors, word):
    if word in word_vectors.stoi:
        return word_vectors[word].numpy()
    else:
        raise NotImplementedError

#---------------------

def getnode(x):
    return wn.synset(x)

def getedges(s):
    dic = {x: i for i, x in enumerate(s)}
    edges = []
    for i, u in enumerate(s):
        for v in u.hypernyms():
            j = dic.get(v)
            if j is not None:
                edges.append((i, j))
    return edges

def induce_parents(s):
    q = s
    vis = set(s)
    l = 0
    while l < len(q):
        u = q[l]
        l += 1
        # for p in u.hypernyms():
        #     if p not in vis:
        #         vis.add(p)
        #         q.append(p)
    return q, [get_synset_embedding(p.name(), word_vectors, get_vector) for p in q]


print('making graph ...')

nodes = list(map(getnode, [_['name'] for _ in json_data['categories']])) # None background

all_nodes, vectors = induce_parents(nodes) # None background
print(len(all_nodes))

edges = getedges(all_nodes) # None background

vectors = torch.from_numpy(np.stack(vectors)).float()


#-----------------------------------------------------------------------------------'
import pickle

new_gcn_trainval = np.ones(len(all_nodes) + 1, dtype=np.int64)
new_gcn_trainval[json_data['source']] = 0
new_gcn_trainval = new_gcn_trainval.tolist()

new_gcn_y = torch.zeros((len(new_gcn_trainval), 2049))
new_gcn_y[json_data['source']] = torch.cat(
    [weight['model']['Box_Outs.cls_score.weight'][json_data['source']],
    weight['model']['Box_Outs.cls_score.bias'][json_data['source']].unsqueeze(1)], 1).cpu()

new_gcn_vectors = torch.zeros(len(new_gcn_trainval), 300)
tmp = vectors
assert (torch.tensor(json_data['word_embeddings']).float() == vectors[:len(json_data['word_embeddings'])]).all()
new_gcn_vectors[0].copy_(tmp.mean(0))
new_gcn_vectors[1:].copy_(tmp)
new_gcn_vectors.copy_(torch.nn.functional.normalize(new_gcn_vectors, dim=1))
# new_gcn_vectors = new_gcn_vectors.numpy()


def pdist2(X,Y):
    return (X.unsqueeze(1) - Y.unsqueeze(0)).norm(2, dim=2)
    # sqrt(sum((reshape(X,3,1,2)-reshape(Y,1,3,2)).^2, 3))
def Compute_Sim(sig_Y, sig_R, Sim_scale, Sim_type):

    if Sim_type == 'RBF_norm':
        # % disp('RBF_norm');
        dist = pdist2(sig_Y, sig_R)
        Sim = (-(dist ** 2) * Sim_scale).exp()
        Sim = Sim / Sim.sum(1, keepdim=True)
    elif Sim_type == 'RBF':
        # % disp('RBF');
        dist = pdist2(sig_Y, sig_R)
        Sim = (-(dist ** 2) * Sim_scale).exp()
    elif Sim_type == 'inner':
        Sim = sig_Y.matmul(sig_R.t())
    elif Sim_type == 'inner_L2':
        Sim = sig_Y.matmul(sig_R.t())
        Sim = Sim / Sim.norm(2, 1, keepdim=True)
    elif Sim_type == 'inner_L1':
        Sim = sig_Y.matmul(sig_R.t())
        Sim = Sim / Sim.norm(1, 1, keepdim=True)
    elif Sim_type == 'inner_positive':
        Sim = sig_Y.matmul(sig_R.t())
        Sim[Sim < 0] = 0
    elif Sim_type == 'inner_L1_positive':
        Sim = sig_Y.matmul(sig_R.t())
        Sim[Sim < 0] = 0
        Sim = Sim / Sim.norm(2, 1, keepdim=True) # Wrong!!!
    elif Sim_type == 'inner_L2_positive':
        Sim = sig_Y.matmul(sig_R.t())
        Sim[Sim < 0] = 0
        Sim = Sim / Sim.norm(1, 1, keepdim=True)
    Sim[torch.isnan(Sim)] = 0; Sim[torch.isinf(Sim)] = 0
    return Sim


Sig_Y = new_gcn_vectors
Sim_type='RBF_norm'
Sim_base = Compute_Sim(Sig_Y[json_data['source'], :], Sig_Y[json_data['source'], :], 1, Sim_type)
Sim_val = Compute_Sim(Sig_Y[json_data['target'], :], Sig_Y[json_data['source'], :], 1, Sim_type)
V = torch.pinverse(Sim_base.double()).float().matmul(new_gcn_y[json_data['source']])

tmp = new_gcn_y.clone()
tmp[json_data['source']] = Sim_base.matmul(V)
tmp[json_data['target']] = Sim_val.matmul(V)

tmp[0].copy_(tmp[1:].mean(0))
tmp.copy_(torch.nn.functional.normalize(tmp, dim=1))

tmp = tmp.cuda()
tmp[json_data['source']] = F.normalize(torch.cat(
    [weight['model']['Box_Outs.cls_score.weight'][json_data['source']],
    weight['model']['Box_Outs.cls_score.bias'][json_data['source']].unsqueeze(1)], 1), dim=1)

# The result doesn't change much because the values are almost identical! so no need for finetune.

# offset = weight['model']['Box_Outs.cls_score.weight'].shape[0]
offset = 1 + len(json_data['source']) + len(json_data['target'])
if 'ignore' in json_data:
    offset += len(json_data['ignore'])

# weight['model']['Box_Outs.cls_score.weight'].copy_(tmp[:offset, :-1])
# weight['model']['Box_Outs.cls_score.bias'].copy_(tmp[:offset, -1])

if args.to_weight is not None:
    weight = torch.load(args.to_weight)

weight['model']['Box_Outs.cls_score.weight'] = tmp[:offset, :-1]
weight['model']['Box_Outs.cls_score.bias'] = tmp[:offset, -1]

if args.to_weight is None:
    torch.save(weight, weight_fn[:-4]+'_sync.pth')
else:
    torch.save(weight, args.to_weight[:-4]+'_sync.pth')

# wordnet 300 2048d,d works better
