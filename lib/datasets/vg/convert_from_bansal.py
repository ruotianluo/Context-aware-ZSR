# coding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
from collections import Counter, defaultdict
import numpy as np
np.random.seed(123)
import re

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--raw_data_dir', default='./data/vg/annotations/', type=str)
parser.add_argument('--output_dir', default='./data/vgbansal/', type=str)
# parser.add_argument('--num_objects', default=600, type=int, help="set to 0 to disable filtering")
parser.add_argument('--num_rels', default=20, type=int, help="set to 0 to disable filtering")
parser.add_argument('--output_json', default='instances_vgbansal', type=str)
parser.add_argument('--prune', default=20, type=int)
parser.add_argument('--drop_source', default=0, type=float)
parser.add_argument('--val', default=0, type=int)


# Process bansal setting
bansal_seen_classes = json.load(open('./data/vg/bansal/vg_seen_classes.json', 'r'))
bansal_unseen_classes = json.load(open('./data/vg/bansal/vg_unseen_classes.json', 'r'))
bansal_synset_dict = json.load(open('./data/vg/bansal/vg_synset_word_dict.json', 'r'))

bansal_invert_dict = {v.lower():k for k,v in bansal_synset_dict.items()}
bansal_seen_synsets =  [bansal_invert_dict[k.lower()] for k in bansal_seen_classes]
bansal_unseen_synsets = [bansal_invert_dict[k.lower()] for k in bansal_unseen_classes]

bansal_train_list = json.load(open('./data/vg/bansal/vg_train_list.json', 'r'))
bansal_train_list = set(['/'.join(_['img_name'].split('/')[-2:]) for _ in bansal_train_list])
bansal_test_list = {'VG_100K_2/'+_ for _ in open('./data/vg/bansal/vg_test_list.txt').read().strip('\n').split('\n')}

args = parser.parse_args()

# Consider all objects
# N_class = args.num_objects if args.num_objects > 0 else None # keep the top 3000 classes
N_relation = args.num_rels if args.num_rels > 0 else None
raw_data_dir = args.raw_data_dir
output_dir = args.output_dir

# ---------------------------------------------------------------------------- #
# Load raw VG annotations and collect top-frequent synsets
# ---------------------------------------------------------------------------- #

with open(raw_data_dir + 'image_data.json') as f:
    raw_img_data = json.load(f)
with open(raw_data_dir + 'objects.json') as f:
    raw_obj_data = json.load(f)
with open(raw_data_dir + 'relationships.json') as f:
    raw_rel_data = json.load(f)

# collect top frequent synsets
all_synsets = {
    synset for img in raw_obj_data
    for obj in img['objects'] for synset in obj['synsets']}
# synset_counter = Counter(all_synsets)
# top_synsets = [
#     synset for synset, _ in synset_counter.most_common(N_class)]

top_synsets = bansal_seen_synsets + bansal_unseen_synsets
# The bansal synset has some mistake. Wrong sense
top_synsets = [_ if _ in all_synsets else _[:-2]+'02' for _ in top_synsets]
# Remove background
top_synsets = [_ for _ in top_synsets if not 'background' in _]
# Make sure bansal synsets are valid
print([_ for _ in top_synsets if not _ in all_synsets])
assert all([_ in all_synsets for _ in top_synsets])

print('number synsets', len(top_synsets))


def clean_string(string):
    # string = string.lower().strip()
    # if len(string) >= 1 and string[-1] == '.':
    #     string = string[:-1].strip()
    # if string[-2:] == ' a':
    #     string = string[:-2]
    # if string[-3:] == ' an':
    #     string = string[:-3]

    predicate = sentence_preprocess(string)
    if predicate in rel_alias_dict:
        predicate = rel_alias_dict[predicate]
    return predicate

import string
def sentence_preprocess(phrase):
    """ preprocess a sentence: lowercase, clean up weird chars, remove punctuation """
    replacements = {
      '½': 'half',
      '—' : '-',
      '™': '',
      '¢': 'cent',
      'ç': 'c',
      'û': 'u',
      'é': 'e',
      '°': ' degree',
      'è': 'e',
      '…': '',
    }
    phrase = phrase.encode('utf-8')
    phrase = phrase.lstrip(' ').rstrip(' ')
    for k, v in replacements.items():
        phrase = phrase.replace(k, v)
    return str(phrase).lower().translate(None, string.punctuation).decode('utf-8', 'ignore')


def preprocess_predicates(data, alias_dict={}):
    for img in data:
        for relation in img['relationships']:
            predicate = sentence_preprocess(relation['predicate'])
            if predicate in alias_dict:
                predicate = alias_dict[predicate]
            relation['predicate'] = predicate


def make_alias_dict(dict_file):
    """create an alias dictionary from a file"""
    out_dict = {}
    vocab = []
    for line in open(dict_file, 'r'):
        alias = line.strip('\n').strip('\r').split(',')
        alias_target = alias[0] if alias[0] not in out_dict else out_dict[alias[0]]
        for a in alias:
            out_dict[a] = alias_target  # use the first term as the aliasing target
        vocab.append(alias_target)
    return out_dict, vocab

rel_alias_dict, _ = make_alias_dict(os.path.join(raw_data_dir, 'relationship_alias.txt'))


def clean_relations(string):
    string = clean_string(string)
    if len(string) > 0:
        return [string]
    else:
        return []


# collect top relation frequent synsets
all_rel_synsets = [
    clean_string(rel['predicate']) for img in raw_rel_data
    for rel in img['relationships']]
rel_synset_counter = Counter(all_rel_synsets)
top_rel_synsets = [
    synset for synset, _ in rel_synset_counter.most_common(N_relation)]
print(top_rel_synsets)

# ---------------------------------------------------------------------------- #
# Make sure synset in wordnet
# ---------------------------------------------------------------------------- #

from nltk.corpus import wordnet as wn
top_synsets = [_ for _ in top_synsets if wn.synset(_) is not None]

# ---------------------------------------------------------------------------- #
# Make sure every word has a word embedding
# ---------------------------------------------------------------------------- #

from torchtext.vocab import GloVe, FastText

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
        return None
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

word_embeddings = []
_top_synsets = []
for synset in top_synsets:
    tmp = get_synset_embedding(synset, word_vectors, get_vector)
    if tmp is not None:
        word_embeddings.append(tmp.tolist())
        _top_synsets.append(synset)
print('%d synsets don\'t have word embedding' %(len(top_synsets) - len(_top_synsets)))
top_synsets = _top_synsets


# ---------------------------------------------------------------------------- #
# Get all embeddings
# ---------------------------------------------------------------------------- #
rel_embeddings = []
# _top_rel_synsets = []
# for synset in top_rel_synsets:
#     tmp = get_synset_embedding(synset, word_vectors, get_vector)
#     if tmp is not None:
#         rel_embeddings.append(tmp.tolist())
#         _top_rel_synsets.append(synset)
# print('%d relationship synsets don\'t have word embedding' %(len(top_rel_synsets) - len(_top_rel_synsets)))
# top_rel_synsets = _top_rel_synsets

# ---------------------------------------------------------------------------- #
# Get all embeddings
# ---------------------------------------------------------------------------- #

embeddings = np.array(word_embeddings)
# normalize the embeddings
embeddings = embeddings / (np.linalg.norm(embeddings, 2, 1, True) + 1e-6)

# # ---------------------------------------------------------------------------- #
# # kmeans embeddings
# # ---------------------------------------------------------------------------- #

# import sklearn.cluster
# km_cluster = sklearn.cluster.KMeans(n_clusters=20)
# result = km_cluster.fit_predict(embeddings)

# # ---------------------------------------------------------------------------- #
# # Sample source and target
# # ---------------------------------------------------------------------------- #

if args.drop_source > 0:
    tmp = np.array(list(range(1, len(bansal_seen_classes)+1-1)))
    np.random.shuffle(tmp)
    source = tmp[:int(len(tmp) * (1-args.drop_source))].tolist()
    if args.val:
        target = tmp[int(len(tmp) * (1-args.drop_source)):].tolist()
    else:
        target = list(range(len(bansal_seen_classes)+1-1, len(top_synsets)+1))
else:
    source = list(range(1, len(bansal_seen_classes)+1-1)) # -1 because original seen includes background
    target = list(range(len(bansal_seen_classes)+1-1, len(top_synsets)+1))


assert len(source) + len(target) == len(top_synsets)

# groups = [[] for _ in range(20)]
# for i, label in enumerate(result):
#     groups[label].append(i)

# for i in range(20):
#     tmp = np.array(groups[i]) + 1 # +1 because 0 is reserved for background
#     np.random.shuffle(tmp)
#     source.extend(tmp[:int(len(tmp) * 0.8)].tolist())
#     target.extend(tmp[int(len(tmp) * 0.8):].tolist())

print('source class number: %d' %(len(source)))
print('target class number: %d' %(len(target)))
print(target)

# ---------------------------------------------------------------------------- #
# build raw "categories"
# ---------------------------------------------------------------------------- #

categories = [
    {'id': (n + 1), 'name': synset} for n, synset in enumerate(top_synsets)]
synset2cid = {c['name']: c['id'] for c in categories}

# ---------------------------------------------------------------------------- #
# build "image"
# ---------------------------------------------------------------------------- #

images = [
    {'id': img['image_id'],
     'width': img['width'],
     'height': img['height'],
     'file_name': img['url'].replace('https://cs.stanford.edu/people/rak248/', ''),
     'coco_id': img['coco_id']}
    for img in raw_img_data]

# Get split

images_train = [_ for _ in images if _['file_name'] in bansal_train_list]
images_val = []
images_test = [_ for _ in images if _['file_name'] in bansal_test_list]

imgids_train = {_['id'] for _ in images_train}
imgids_val = {}
imgids_test = {_['id'] for _ in images_test}


# ---------------------------------------------------------------------------- #
# build "annotations"
# ---------------------------------------------------------------------------- #

annotations = []
skip_count_1, skip_count_2, skip_count_3 = 0, 0, 0
for img in raw_obj_data:
    for obj in img['objects']:
        synsets = obj['synsets']
        if len(synsets) == 0:
            skip_count_1 += 1
        elif len(synsets) > 1:
            skip_count_2 += 1
        elif synsets[0] not in synset2cid:
            skip_count_3 += 1
        else:
            cid = synset2cid[synsets[0]]
            bbox = [obj['x'], obj['y'], obj['w'], obj['h']]
            area = obj['w'] * obj['h']
            ann = {'id': obj['object_id'],
                   'image_id': img['image_id'],
                   'category_id': cid,
                   'segmentation': [],
                   'area': area,
                   'bbox': bbox,
                   'iscrowd': 0,
                   'object_id': obj['object_id']}
            annotations.append(ann)

# ---------------------------------------------------------------------------- #
# Filter away images that have no objects
# ---------------------------------------------------------------------------- #

from collections import defaultdict
obj_in_img_count = defaultdict(lambda: 0)
for ann in annotations:
    obj_in_img_count[ann['image_id']] += 1
print('Object numbers in image histogram:')
print(np.histogram(np.array(list(obj_in_img_count.values())), bins=20, range=(1,20))[0].tolist())
_images = [_ for _ in images if obj_in_img_count[_['id']] > 0]
print('%d images have no object' %(len(images) - len(_images)))
iamges = _images

# ---------------------------------------------------------------------------- #
# Only keep the predicates that appear in the source classes
# ---------------------------------------------------------------------------- #

keep_rel_synsets = []
for img in raw_rel_data:
    for rel in img['relationships']:
        synsets = rel['object']['synsets']
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in synset2cid:
            continue
        synsets = rel['subject']['synsets']
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in synset2cid:
            continue
        synsets = clean_relations(rel['predicate'])
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in top_rel_synsets:
            continue
        if synset2cid[rel['object']['synsets'][0]] - 1 in source and\
                synset2cid[rel['subject']['synsets'][0]] - 1 in source:
            keep_rel_synsets.append(synsets[0])

_top_rel_synsets = [_ for _ in top_rel_synsets if _ in keep_rel_synsets]
print('%d predicates only appear in target classes and not in source classes' %(len(top_rel_synsets) - len(_top_rel_synsets)))
top_rel_synsets = _top_rel_synsets

# ---------------------------------------------------------------------------- #
# build raw "relationship categories"
# ---------------------------------------------------------------------------- #

rel_categories = [
    {'id': (n + 1), 'name': synset} for n, synset in enumerate(top_rel_synsets)]
rel_synset2rid = {r['name']: r['id'] for r in rel_categories}

# ---------------------------------------------------------------------------- #
# Build knowledge graph annotation
# ---------------------------------------------------------------------------- #

real_relationships_anno = {}
skip_count_1, skip_count_2, skip_count_3 = 0, 0, 0
for img in raw_rel_data:
    real_relationships_anno[img['image_id']] = []
    for rel in img['relationships']:
        synsets = rel['object']['synsets']
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in synset2cid:
            continue
        synsets = rel['subject']['synsets']
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in synset2cid:
            continue
        synsets = clean_relations(rel['predicate'])
        if len(synsets) == 0:
            skip_count_1 += 1
        elif  len(synsets) > 1:
            skip_count_2 += 1
        elif synsets[0] not in rel_synset2rid:
            skip_count_3 += 1
        else:
            anno = {}
            anno['object_id'] = rel['object']['object_id']
            anno['subject_id'] = rel['subject']['object_id']
            anno['rel_id'] = rel_synset2rid[synsets[0]]
            real_relationships_anno[img['image_id']].append(anno)

# ---------------------------------------------------------------------------- #
# Build knowledge graph annotation
# ---------------------------------------------------------------------------- #

relationships = defaultdict(lambda:0)
skip_count_1, skip_count_2, skip_count_3 = 0, 0, 0
for img in raw_rel_data:
    # if not img['image_id'] in imgids_train:
    #     continue
    for rel in img['relationships']:
        synsets = rel['object']['synsets']
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in synset2cid:
            continue
        synsets = rel['subject']['synsets']
        if len(synsets) == 0 or len(synsets) > 1 or synsets[0] not in synset2cid:
            continue
        synsets = clean_relations(rel['predicate'])
        if len(synsets) == 0:
            skip_count_1 += 1
        elif  len(synsets) > 1:
            skip_count_2 += 1
        elif synsets[0] not in rel_synset2rid:
            skip_count_3 += 1
        else:
            relationships[(rel['subject']['synsets'][0], rel['object']['synsets'][0], synsets[0])] += 1

relationships_anno = []
# import pdb;pdb.set_trace()
# tmp = defaultdict(dict)
# for rel in sorted(relationships.keys(), key=lambda _:-relationships[_]): tmp[(rel[0], rel[1])][rel[2]] = relationships[rel]
for rel in sorted(relationships.keys(), key=lambda _:-relationships[_]):
    if relationships[rel] < args.prune :
        continue
    relationships_anno.append({
        'subject_id': synset2cid[rel[0]],
        'object_id': synset2cid[rel[1]],
        'rel_id': rel_synset2rid[rel[2]],
        'count': relationships[rel]})

# ---------------------------------------------------------------------------- #
# Save to json file
# ---------------------------------------------------------------------------- #

if not os.path.isdir(output_dir):
    os.makedirs(output_dir)

with open(output_dir + args.output_json + '_raw.json', 'w') as f:
    dataset_vg = {
        'images': images,
        'annotations': annotations,
        'categories': categories,
        'word_embeddings': word_embeddings,
        'source': source,
        'target': target,
        'rel_embeddings': rel_embeddings,
        'relationships': relationships_anno,
        'real_relationships': real_relationships_anno,
        'rel_categories': rel_categories}
    json.dump(dataset_vg, f)



# ---------------------------------------------------------------------------- #
# Split into train, val and test
# ---------------------------------------------------------------------------- #

# imgids_train = {_['id'] for _ in images[:-5000]}
# imgids_val = {}
# imgids_test = {_['id'] for _ in images[-5000:]}

# images_train = images[:-5000]
# images_val = []
# images_test = images[-5000:]


print('number of train, val, test images: {}, {}, {}'.format(
    len(images_train), len(images_val), len(images_test)))

# Save the dataset splits
annotations_train = [
    ann for ann in dataset_vg['annotations']
    if ann['image_id'] in imgids_train]
annotations_val = [
    ann for ann in dataset_vg['annotations']
    if ann['image_id'] in imgids_val]
annotations_test = [
    ann for ann in dataset_vg['annotations']
    if ann['image_id'] in imgids_test]

# ---------------------------------------------------------------------------- #
# Save to json file
# ---------------------------------------------------------------------------- #

import copy
dataset_vg_train = copy.deepcopy(dataset_vg)
dataset_vg_train.update({
    'images': images_train,
    'annotations': annotations_train})
dataset_vg_val = copy.deepcopy(dataset_vg)
dataset_vg_val.update({
    'images': images_val,
    'annotations': annotations_val})
dataset_vg_test = copy.deepcopy(dataset_vg)
dataset_vg_test.update({
    'images': images_test,
    'annotations': annotations_test})


def filter_annotations(ds, func):
    ds = copy.deepcopy(ds)
    ds.update({'annotations': func(ds['annotations'])})
    return ds

with open(output_dir + '%s_train.json' %(args.output_json), 'w') as f:
    json.dump(dataset_vg_train, f)
with open(output_dir + '%s_val.json' %(args.output_json), 'w') as f:
    json.dump(dataset_vg_val, f)
with open(output_dir + '%s_test.json' %(args.output_json), 'w') as f:
    json.dump(dataset_vg_test, f)


with open(output_dir + '%s_source_test.json' %(args.output_json), 'w') as f:
    json.dump(filter_annotations(dataset_vg_test,\
        lambda x: [_ for _ in x if _['category_id'] in source]), f)
with open(output_dir + '%s_target_test.json' %(args.output_json), 'w') as f:
    json.dump(filter_annotations(dataset_vg_test,\
        lambda x: [_ for _ in x if _['category_id'] in target]), f)