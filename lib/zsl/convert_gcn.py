# problem, during test, the corresp should also be changed..
import argparse

from nltk.corpus import wordnet as wn
import re

import torch
import pickle
import json
import numpy as np
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='vgbansal', type=str, help='vgzs')
parser.add_argument('--weight_folder', default='./Outputs/iter_reason_new/rluo_irnb_vg_gt_cooc_pgeo_sc_step', type=str)
parser.add_argument('--hiddens', default='--hiddens 2048d,d', type=str)

args = parser.parse_args()

# vg_dataset = 'vgzs'
vg_dataset = args.dataset

json_data = json.load(open('./data/%s/instances_%s_raw.json' %(vg_dataset, vg_dataset)))
weight_fn = args.weight_folder + '/ckpt/model_final.pth'
output_dir = './externals/zsl-gcn-pth/data/%s_wn' %(args.weight_folder.split('/')[-1])

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
        for p in u.hypernyms():
            if p not in vis:
                vis.add(p)
                q.append(p)
    return q, [get_synset_embedding(p.name(), word_vectors, get_vector) for p in q]


print('making graph ...')

nodes = list(map(getnode, [_['name'] for _ in json_data['categories']])) # None background

all_nodes, vectors = induce_parents(nodes) # None background
print(len(all_nodes))

edges = getedges(all_nodes) # None background

vectors = torch.from_numpy(np.stack(vectors)).float()


#-----------------------------------------------------------------------------------'
import pickle


# gcn_graph = pickle.load(open('./externals/zsl-gcn-pth/data/glove_res50/ind.NELL.graph', 'r'))
# gcn_vectors = pickle.load(open('./externals/zsl-gcn-pth/data/glove_res50/ind.NELL.allx_dense', 'r'))
# gcn_y = pickle.load(open('./externals/zsl-gcn-pth/data/glove_res50/ind.NELL.ally_multi', 'r'))
# gcn_trainval = pickle.load(open('./externals/zsl-gcn-pth/data/glove_res50/ind.NELL.index', 'r'))
# gcn_index = json.load(open('./externals/zsl-gcn-pth/data/list/invdict_wordn.json'))

new_gcn_trainval = np.ones(len(all_nodes) + 1, dtype=np.int64)
new_gcn_trainval[json_data['source']] = 0
new_gcn_trainval = new_gcn_trainval.tolist()

new_gcn_y = np.zeros((len(new_gcn_trainval), 2049), dtype=np.float32)
new_gcn_y[json_data['source']] = torch.cat(
    [weight['model']['Box_Outs.cls_score.weight'][json_data['source']],
    weight['model']['Box_Outs.cls_score.bias'][json_data['source']].unsqueeze(1)], 1).cpu().numpy()

new_gcn_vectors = torch.zeros(len(new_gcn_trainval), 300)
tmp = vectors
assert (torch.tensor(json_data['word_embeddings']).float() == vectors[:len(json_data['word_embeddings'])]).all()
new_gcn_vectors[0].copy_(tmp.mean(0))
new_gcn_vectors[1:].copy_(tmp)
new_gcn_vectors.copy_(torch.nn.functional.normalize(new_gcn_vectors, dim=1))
new_gcn_vectors = new_gcn_vectors.numpy()

new_gcn_graph = {_:[] for _ in range(len(new_gcn_trainval))}
for edge in edges:
    new_gcn_graph[edge[0]+1].append(edge[1]+1) # +1 because of background
    new_gcn_graph[edge[1]+1].append(edge[0]+1)

new_gcn_graph = {_:list(set(new_gcn_graph[_])) for _ in new_gcn_graph}

print([len(_) for _ in new_gcn_graph.values()])

import os

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

with open(output_dir + '/ind.NELL.graph', 'w') as f:
    pickle.dump(new_gcn_graph, f)

with open(output_dir + '/ind.NELL.allx_dense', 'w') as f:
    pickle.dump(new_gcn_vectors, f)

with open(output_dir + '/ind.NELL.ally_multi', 'w') as f:
    pickle.dump(new_gcn_y, f)

with open(output_dir + '/ind.NELL.index', 'w') as f:
    pickle.dump(new_gcn_trainval, f)


tmp = output_dir.split('/')[-1]
os.system('cd ./externals/zsl-gcn-pth/src;python gcn/train_gcn.py ' + \
    '--dataset ../data/%s/ %s ' %(tmp, args.hiddens) +\
    '--save_path log_detectron/log_%s ' %tmp +\
    '--adj_norm_type in --feat_norm_type l2 --epochs 350')

tmp = torch.from_numpy(np.load('./externals/zsl-gcn-pth/src/log_detectron/log_%s//feat__300' %(tmp))).cuda()

tmp[json_data['source']] = F.normalize(torch.cat(
    [weight['model']['Box_Outs.cls_score.weight'][json_data['source']],
    weight['model']['Box_Outs.cls_score.bias'][json_data['source']].unsqueeze(1)], 1), dim=1)

# The result doesn't change much because the values are almost identical! so no need for finetune.

# offset = weight['model']['Box_Outs.cls_score.weight'].shape[0]
offset = 1 + len(json_data['source']) + len(json_data['target'])

# weight['model']['Box_Outs.cls_score.weight'].copy_(tmp[:offset, :-1])
# weight['model']['Box_Outs.cls_score.bias'].copy_(tmp[:offset, -1])

weight['model']['Box_Outs.cls_score.weight'] = tmp[:offset, :-1]
weight['model']['Box_Outs.cls_score.bias'] = tmp[:offset, -1]

torch.save(weight, weight_fn[:-4]+'_gcn_wn.pth')

# wordnet 300 2048d,d works better
