
""" Graph conv blocks.

Implementation based on https://github.com/cvlab-epfl/voxel2mesh.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pytorch3d.ops as py3dnn
from torch_sparse import SparseTensor

from utils.custom_layers import IdLayer
from utils.utils import Euclidean_weights
from logger import measure_time


class GraphConvNorm(py3dnn.GraphConv):
    """ Wrapper for pytorch3d.ops.GraphConv that normalizes the features
    w.r.t. the degree of the vertices.
    """
    def __init__(self, input_dim: int, output_dim: int, init: str='normal',
                 directed: bool=False, **kwargs):
        super().__init__(input_dim, output_dim, init, directed)
        if kwargs.get('weighted_edges', False) is True:
            raise ValueError(
                "pytorch3d.ops.GraphConv cannot be edge-weighted."
            )

        # Make sure bias is initialized with 0
        self.w0.bias.data.zero_()
        self.w1.bias.data.zero_()

    @measure_time
    def forward(self, verts, edges):
        # Normalize with 1 + N(i)
        # Attention: This requires the edges to be unique!
        D_inv = 1.0 / (1 + torch.unique(edges, return_counts=True)[1].unsqueeze(1))
        return D_inv * super().forward(verts, edges)


class SparseGraphConv(gnn.GraphConv):
    """ Graph conv based on sparse matrix multiplications. This is very similar
    to pytorch3d graph convs with the difference that a bias is only learned
    for the linear layer operating on the neighbor's features.
    """
    def __init__(self, input_dim: int, output_dim: int, init: str='normal',
                 directed: bool=False, **kwargs):
        super().__init__(input_dim, output_dim)
        if kwargs.get('weighted_edges', False) is True:
            raise NotImplementedError("Could be implemented.")

        self._directed = directed

        # Set at first forward pass
        self._adj_t = None
        self._D_inv = None

        if init == 'normal':
            self._init_normal()
        elif init == 'zero':
            self._init_zero()
        else:
            raise NotImplementedError("Only normal and zero init supported.")

    def _init_normal(self):
        """ Same init as pytorch3d graph convs. """
        nn.init.normal_(self.lin_rel.weight, mean=0, std=0.01)
        nn.init.normal_(self.lin_root.weight, mean=0, std=0.01)
        self.lin_rel.bias.data.zero_()

    def _init_zero(self):
        self.lin_rel.weight.data.zero_()
        self.lin_root.weight.data.zero_()
        self.lin_rel.bias.data.zero_()

    def _set_adj_t(self, edges, n_verts):
        assert edges.shape[1] == 2, "Assumes edges of shape [n_edges, 2]"
        if not self._directed:
            # Unique bidirectional edges
            edges = torch.unique(
                torch.cat([edges, edges.flip(dims=[1])], dim=0),
                dim=0
            )
        self._adj_t = SparseTensor(
            row=edges[:, 0], col=edges[:, 1], sparse_sizes=(n_verts, n_verts)
        ).t()

        self._D_inv = 1.0 / (1 + torch.unique(edges, return_counts=True)[1].unsqueeze(1))

        # Sanity
        if not self._directed:
            assert self._adj_t == self._adj_t.t()

    @measure_time
    def forward(self, verts, edges):
        if self._adj_t is None:
            self._set_adj_t(edges, verts.shape[0])

        return self._D_inv * super().forward(verts, self._adj_t)


class LinearLayer(nn.Linear):
    """ A linear layer replacing graph convs.
    """
    def __init__(self, input_dim: int, output_dim: int, init='normal',
                 weighted_edges=None):
        super().__init__(input_dim, output_dim)

        # Init weight either normal or zero, bias always zero
        if init == 'normal':
            nn.init.normal_(self.weight, mean=0.0, std=0.01)
        elif init == 'zero':
            self.weight.data.zero_()
        else:
            raise NotImplementedError("Only normal and zero init supported.")
        self.bias.data.zero_()

    def forward(self, verts, edges):
        # Linear layer ignores edges
        return super().forward(verts)


class Features2FeaturesResidual(nn.Module):
    """ A residual graph conv block consisting of 'hidden_layer_count' many graph convs """

    def __init__(self, in_features, out_features, hidden_layer_count,
                 norm='batch', GC=py3dnn.GraphConv, p_dropout=None, weighted_edges=False):
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

        # Optional dropout layer
        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = IdLayer()

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
                # Conv --> Norm --> ReLU (--> Dropout)
                features = self.dropout(F.relu(nl(gconv(features, edges))))

        return features


def zero_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.weight, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, py3dnn.GraphConv):
        # Bug in GraphConv: bias is not initialized to zero
        nn.init.constant_(m.w0.weight, 0.0)
        nn.init.constant_(m.w0.bias, 0.0)
        nn.init.constant_(m.w1.weight, 0.0)
        nn.init.constant_(m.w1.bias, 0.0)
    else:
        pass
