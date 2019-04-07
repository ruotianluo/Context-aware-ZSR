import torch


# Copy from https://github.com/msracver/Relation-Networks-for-Object-Detection/blob/master/relation_rcnn/operator_py/learn_nms.py
def extract_multi_position_matrix_nd(bbox):
    xmin, ymin, xmax, ymax = torch.split(bbox, 1, 1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    # [num_fg_classes, num_boxes, num_boxes]
    delta_x = center_x - center_x.t()
    delta_x = delta_x / bbox_width
    # delta_x = nd.log(nd.maximum(nd.abs(delta_x), 1e-3))

    delta_y = center_y - center_y.t()
    delta_y = delta_y / bbox_height
    # delta_y = nd.log(nd.maximum(nd.abs(delta_y), 1e-3))

    delta_width = bbox_width / bbox_width.t()
    # delta_width = nd.log(delta_width)

    delta_height = bbox_height / bbox_height.t()
    # delta_height = nd.log(delta_height)
    position_matrix = torch.stack([delta_x, delta_y, delta_width, delta_height], 2)
    position_matrix = position_matrix.abs().clamp(min=1e-3).log()
    return position_matrix

def extract_pairwise_multi_position_embedding_nd(position_mat, feat_dim=64, wave_length=1000):
    """ Extract multi-class position embedding
    Args:
        position_mat: [num_rois, num_rois, 4]
        feat_dim: dimension of embedding feature
        wave_length:
    Returns:
        embedding: [num_rois, num_rois, feat_dim]
    """
    feat_range = torch.arange(0, feat_dim / 8).type_as(position_mat)
    dim_mat = torch.pow(position_mat.new_full((1,), wave_length),
                          (8. / feat_dim) * feat_range)
    dim_mat = dim_mat.reshape(1, 1, 1, -1)
    position_mat = (100.0 * position_mat).unsqueeze(3)
    div_mat = position_mat / dim_mat
    sin_mat = torch.sin(div_mat)
    cos_mat = torch.cos(div_mat)
    # embedding, [num_rois, num_rois, 4, feat_dim/4]
    embedding = torch.cat([sin_mat, cos_mat], dim=3)
    embedding = embedding.reshape(*(embedding.shape[:2] + (feat_dim,)))
    return embedding

# In relation network, the rest of the network is a fully connected nn.Linear(64, 16) + relu


def get_chw(bbox):
    # given bbox, output center, height and width
    xmin, ymin, xmax, ymax = torch.split(bbox, 1, 1)
    # [num_fg_classes, num_boxes, 1]
    bbox_width = xmax - xmin + 1.
    bbox_height = ymax - ymin + 1.
    center_x = 0.5 * (xmin + xmax)
    center_y = 0.5 * (ymin + ymax)
    return center_x, center_y, bbox_width, bbox_height

def get_pairwise_feat(boxA, boxB):
    # Generate 6d feature given two (batches of ) boxes 
    xA, yA, wA, hA = get_chw(boxA)
    xB, yB, wB, hB = get_chw(boxB)
    feat = torch.cat([
        (xA - xB) / wA,
        (yA - yB) / hA,
        (wA.log() - wB.log()),
        (hA.log() - hB.log()),
        (xB - xA) / wB,
        (yB - yA) / hB], 1)
    return feat

def get_proposal_feat(rois):
    # Enumerate all roi combinations, and output the features
    roisS = rois.unsqueeze(1).expand(-1, rois.shape[0], -1).reshape(-1, 4)
    roisO = rois.unsqueeze(0).expand(rois.shape[0], -1, -1).reshape(-1, 4)
    roisP = torch.stack([
        torch.min(roisS[:, 0], roisO[:, 0]),
        torch.min(roisS[:, 1], roisO[:, 1]),
        torch.max(roisS[:, 2], roisO[:, 2]),
        torch.max(roisS[:, 3], roisO[:, 3])
    ], 1)
    feat = torch.cat([
        get_pairwise_feat(roisS, roisO),
        get_pairwise_feat(roisS, roisP),
        get_pairwise_feat(roisO, roisP)], 1)
    return feat

# From Relationships Proposal networks its followed by two consecutive fc layers with 64 outputs.
# (Can't understand)

if __name__ == '__main__':
    # test.
    input = torch.rand(10, 4)
    out = extract_multi_position_matrix_nd(input)
    extract_pairwise_multi_position_embedding_nd(out, 128)
    tmp = get_proposal_feat(input)
    print(input)

