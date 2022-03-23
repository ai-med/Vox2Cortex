
""" Aggregation of voxel features at vertex locations """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.ops import knn_points

from utils.utils import int_to_binlist

def aggregate_trilinear(voxel_features, vertices, mode='bilinear'):
    """ Trilinear/bilinear aggregation of voxel features at vertex locations """
    if vertices.shape[-1] == 3:
        vertices_ = vertices[:, :, None, None]
    elif vertices.shape[-1] == 2:
        vertices_ = vertices[:, :, None]
    else:
        raise ValueError("Wrong dimensionality of vertices.")
    features = F.grid_sample(voxel_features, vertices_, mode=mode,
                             padding_mode='border', align_corners=True)
    # Channel dimension <--> V dimension
    if vertices.shape[-1] == 3:
        features = features[:, :, :, 0, 0].transpose(2, 1)
    else: # 2D
        features = features[:, :, :, 0].transpose(2, 1)

    return features

def aggregate_from_indices(voxel_features, vertices, skip_indices,
                           mode='bilinear'):
    """ Aggregation of voxel features at different encoder/decoder indices """
    features = []
    for i in skip_indices:
        if mode == 'bilinear' or mode == 'trilinear':
            # Trilinear = bilinear
            mode = 'bilinear' if mode == 'trilinear' else 'bilinear'
            features.append(aggregate_trilinear(
                voxel_features[i], vertices, mode
            ))

    return torch.cat(features, dim=2)

def aggregate_structural_features(coords: torch.Tensor,
                                  group_idx: Tuple[Tuple[int]],
                                  exchange_coords: bool,
                                  K: int=5):
    """ Aggregation of structural features. For a vertex v, this includes
    a structural encoding related to the structure group to which v belongs
    and positions of k nearby vertices belonging to different structures
    than v. Groups are required to be of equal size!
    """
    features = []
    device = coords.device
    B, M, V, D = coords.shape

    # Number of groups
    G = len(group_idx)
    if G <= 1:
        raise ValueError("Number of groups should be > 1.")

    # Number of digits for structural encoding
    n_digits = int(np.ceil(np.log2(G)))

    for gi, idx in enumerate(group_idx):
        # Get vertices of structure and treat as one pointcloud
        p1 = coords[:, idx, :, :].view(B, -1, D)

        # Add structural encoding, aka 'surface id'
        struct_encoding = torch.tensor(
            int_to_binlist(gi, n_digits)
        ).repeat(B, len(idx) * V, 1).to(device)

        if exchange_coords:
            # Treat vertices of all other structures as one pointcloud
            p2 = coords[:, tuple(set(range(M)) - set(idx)), :, :].view(B, -1, D)

            # Get nearest neighbors
            _, _, nn_features = knn_points(p1, p2, K=K, return_nn=True)
            nn_features = nn_features.view(B, -1, K * D)

            nn_features = torch.cat((nn_features, struct_encoding), dim = 2)

        else:
            nn_features = struct_encoding

        features.append(nn_features.view(B, len(idx), V, -1))

    features = torch.cat(features, dim=1)

    # Correct order
    order = np.argsort(np.array(group_idx).flatten())
    features = features[:, order, :, :]

    return features
