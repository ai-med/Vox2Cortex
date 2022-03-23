
""" Graph conv blocks for Vox2Cortex.

Implementation based on https://github.com/cvlab-epfl/voxel2mesh.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@tum.de"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv

from utils.utils_vox2cortex.custom_layers import IdLayer
from utils.utils import Euclidean_weights


class GraphConvNorm(GraphConv):
    """ Wrapper for pytorch3d.ops.GraphConv that normalizes the features
    w.r.t. the degree of the vertices.
    """
    def __init__(self, input_dim: int, output_dim: int, init: str='normal',
                 directed: bool=False, **kwargs):
        super().__init__(input_dim, output_dim, init, directed)
        if kwargs.get('weighted_edges', False) == True:
            raise ValueError(
                "pytorch3d.ops.GraphConv cannot be edge-weighted."
            )

    def forward(self, verts, edges):
        # Normalize with 1 + N(i)
        # Attention: This requires the edges to be unique!
        D_inv = 1.0 / (1 + torch.unique(edges, return_counts=True)[1].unsqueeze(1))
        return D_inv * super().forward(verts, edges)


class Features2FeaturesResidual(nn.Module):
    """ A residual graph conv block consisting of 'hidden_layer_count' many graph convs """

    def __init__(self, in_features, out_features, hidden_layer_count,
                 norm='batch', GC=GraphConv, weighted_edges=False):
        assert norm in ('none', 'layer', 'batch'), "Invalid norm."

        super().__init__()

        self.out_features = out_features

        self.gconv_first = GC(in_features, out_features, weighted_edges=weighted_edges)
        if norm == 'batch':
            self.norm_first = nn.BatchNorm1d(out_features)
        elif norm == 'layer':
            self.norm_first = nn.LayerNorm(out_features)
        else: # none
            self.norm_first = IdLayer()

        gconv_hidden = []
        for _ in range(hidden_layer_count):
            # No weighted edges and no propagated coordinates in hidden layers
            gc_layer = GC(out_features, out_features, weighted_edges=False)
            if norm == 'batch':
                norm_layer = nn.BatchNorm1d(out_features)
            elif norm == 'layer':
                norm_layer = nn.LayerNorm(out_features)
            else: # none
                norm_layer = IdLayer() # Id

            gconv_hidden += [nn.Sequential(gc_layer, norm_layer)]

        self.gconv_hidden = nn.Sequential(*gconv_hidden)

    def forward(self, features, edges):
        if features.shape[-1] == self.out_features:
            res = features
        else:
            res = F.interpolate(features.unsqueeze(1), self.out_features,
                                mode='nearest').squeeze(1)

        # Conv --> Norm --> ReLU
        features = F.relu(self.norm_first(self.gconv_first(features, edges)))
        for i, (gconv, nl) in enumerate(self.gconv_hidden, 1):
            if i == len(self.gconv_hidden):
                # Conv --> Norm --> Addition --> ReLU
                features = F.relu(nl(gconv(features, edges)) + res)
            else:
                # Conv --> Norm --> ReLU
                features = F.relu(nl(gconv(features, edges)))

        return features


def zero_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, GraphConv):
        # Bug in GraphConv: bias is not initialized to zero
        nn.init.constant_(m.w0.weight, 0.0)
        nn.init.constant_(m.w0.bias, 0.0)
        nn.init.constant_(m.w1.weight, 0.0)
        nn.init.constant_(m.w1.bias, 0.0)
    else:
        pass
