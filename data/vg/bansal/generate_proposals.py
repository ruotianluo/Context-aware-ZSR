from six.moves import cPickle as pickle

x = pickle.load(open('template.pkl', 'rb'))

import json
test_images = json.load(open('../../vgbansal/instances_vgbansal_test.json', 'r'))

ids = [_['id'] for _ in test_images['images']]

x['ids'] = ids

import torch
raw_proposals = torch.load('raw_proposals.pth')

raw_proposals = {k:v[v[:,-1]>0.07] for k,v in raw_proposals.items()}

import numpy as np
def xywh_to_xyxy(xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format."""
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')

x['boxes'] = [xywh_to_xyxy(raw_proposals[str(_)][:, :4]) for _ in ids]
x['scores'] = [raw_proposals[str(_)][:, -1] for _ in ids] 

pickle.dump(x, open('bansal_test_proposals.pkl', 'wb')) 
